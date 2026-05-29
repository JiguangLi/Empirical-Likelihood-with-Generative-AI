"""
- Produces:
    (1) batch_input_*.jsonl  (requests to submit)
    (2) batch_manifest.jsonl (custom_id -> company + obs specs)
    (3) batch_output_<id>.jsonl (downloaded raw batch outputs)
    (4) synthetic_enriched.jsonl (one firm-date per line, with final_prob + label)
    (5) synthetic_firmdate.csv (your firm-date CSV format)
    (6) synthetic_bad_rows.jsonl (invalid rows / errors)
    (7) token_usage_summary.json (token totals; optional cost estimate)

Typical usage
-------------
# 1) Prepare+Submit batch
python generate_synthetic_news_batch.py \
  --companies_csv companies.csv \
  --rows_per_company 10000 \
  --obs_per_request 10 \
  --out_dir ./synthetic_out \
  --stage submit

# 2) Poll
python generate_synthetic_news_batch.py --out_dir ./synthetic_out --stage poll

# 3) Download
python generate_synthetic_news_batch.py --out_dir ./synthetic_out --stage download

# 4) Parse -> enriched JSONL + CSV
python generate_synthetic_news_batch.py --out_dir ./synthetic_out --stage parse \
  --alpha 1.0 --p0 0.52 --flip_rate 0.25

Notes
-----
- Batch API limits: up to 50,000 requests per batch job and input file up to 200MB.
- With obs_per_request=10, 400k firm-date obs => 40k requests (fits in one batch), BUT file size must still be <200MB.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

# -----------------------------
# Source labels (short codes)
# -----------------------------
NEWS_SOURCES: List[str] = ["WSJ", "FT", "BBG", "DJN", "MW", "BRN"] 
PR_SOURCES: List[str] = ["PRN", "BW", "GNW", "Cision", "Pressetext", "CNW"] 

# -----------------------------
# system prompt
# -----------------------------
SYSTEM_PROMPT: str = """You generate synthetic, fictional overnight financial news-feed headlines for large-cap US equities (circa 2025).

Goal: mimic a noisy real-world market-news feed. Do NOT write “explain-the-return” stories.
Do NOT mention “prediction”, “tomorrow’s return”, “expected to rise/fall”, or anything about modeling.
Do NOT copy, paraphrase, or closely imitate any real headline. Invent everything.

INPUT JSON:
{ c: company_name, obs: [ { d: YYYY-MM-DD, k: 1..25, pr: 0..k }, ... ] }

For each observation (date d):
- Produce exactly k DISTINCT headline items in a realistic mix of styles and signal-to-noise.
- Exactly pr items must be press-release style (p=1) and MUST use PR sources.
- The remaining items must be regular newswire/editorial style (p=0) and MUST use NEWS sources.

Realism constraints (match typical vendor-normalized feeds):
1) casing/format: output headline text in lower-case. Vary structure; fragments are OK.
2) length: vary length widely. Include some short (4–8 words), many medium (9–14), and some long (18–30).
3) numbers: roughly half of headlines should contain at least one number (%, $, share counts, price targets, form numbers like 8-k/10-q/form 4, or times like “5 pm et”).
4) source reuse: per observation, reuse sources heavily (like real feeds). Usually 1 dominant NEWS source (often DJN) + 0–2 additional NEWS sources.
   For PR items, use at most 1–2 PR sources total.
5) topic mix (approximate, NOT exact):
   - 10–20% analyst/ratings items (upgrade/downgrade/price target/maintained at; include bank names).
   - 5–10% filings/insider/administrative items (files 8-k, form 4, registers shares, dividend record date, etc.).
   - 5–10% market roundup items that mention this company among a list of other large-cap tickers (movers, “stocks to watch”, futures headlines, etc.).
   - remaining items: ordinary company/industry news, product notes, litigation/regulatory, macro/sector pieces, conferences.
   Many items should be neutral or only weakly informative; mixed tone in the same night is normal.
6) relevance score r (51..100): mimic real feeds—mostly 85–100, sometimes 70–84, rarely 51–69.
7) tone z: output net overnight tone z in [-2, 2] summarizing the whole set (bullish positive, bearish negative).
   Most dates should be near-neutral (|z| <= 0.7). Extreme values are rare and should correspond to clear catalysts.

Allowed NEWS sources (use code exactly): BBG, FT, WSJ, DJN, MW, BRN
Allowed PR sources (use code exactly): PRN, BW, GNW, Cision, Pressetext, CNW

OUTPUT JSON (and nothing else):
{
  "x": [
    {
      "d": "...",
      "z": ...,
      "h": [
        {"t": "...", "s": "...", "r": 51..100, "p": 0/1},
        ...
      ]
    },
    ...
  ]
}
"""

def response_schema(max_obs_per_request: int) -> Dict[str, Any]:
    """JSON schema for Structured Outputs (compact keys)."""
    all_sources = NEWS_SOURCES + PR_SOURCES
    return {
        "type": "object",
        "properties": {
            "x": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_obs_per_request,
                "items": {
                    "type": "object",
                    "properties": {
                        "d": {"type": "string"},    # date
                        "z": {"type": "number", "minimum": -2.0, "maximum": 2.0},
                        "h": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 25,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "t": {"type": "string"}, # text
                                    "s": {"type": "string", "enum": all_sources},
                                    "r": {"type": "integer", "minimum": 51, "maximum": 100},
                                    "p": {"type": "integer", "enum": [0, 1]},
                                },
                                "required": ["t", "s", "r", "p"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["d", "z", "h"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["x"],
        "additionalProperties": False,
    }

# -----------------------------
# Helpers
# -----------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

# -----------------------------
# Sampling: K and PR count
# -----------------------------
def sample_num_headlines(rng: np.random.Generator, max_k: int = 25) -> int:
    """
    Right-skewed K in [1..max_k] tuned to look like a real overnight news feed:
      - many 1–5 headline nights
      - occasional 10–20 headline nights
      - a small fraction of very-high-volume nights (wire floods) that hit the cap

    """
    # 3-component Poisson mixture
    w_small = 0.72      # routine nights
    w_medium = 0.18     # busier nights (earnings, multi-story coverage)
    # remaining probability goes to "overflow" nights that often clip at max_k
    lam_small = 3.0
    lam_medium = 13.0
    lam_overflow = 60.0

    u = rng.random()
    if u < w_small:
        k = rng.poisson(lam_small)
    elif u < (w_small + w_medium):
        k = rng.poisson(lam_medium)
    else:
        k = rng.poisson(lam_overflow)

    return int(np.clip(k, 1, max_k))

def sample_num_press_releases(rng: np.random.Generator, k: int) -> int:
    """
    Real feeds tend to have a higher PR share on low-headline nights and a lower PR share
    on high-headline nights (wire volume dominates). Use a simple piecewise rate.
    """
    if k <= 3:
        pr_rate = 0.26
    elif k <= 5:
        pr_rate = 0.23
    elif k <= 10:
        pr_rate = 0.18
    elif k <= 20:
        pr_rate = 0.14
    else:
        pr_rate = 0.10

    pr = int(rng.binomial(n=int(k), p=float(pr_rate)))
    return int(np.clip(pr, 0, k))

def random_2025_business_day(rng: np.random.Generator) -> str:
    # Business days only
    days = pd.date_range("2025-01-02", "2025-12-31", freq="B")
    day = days[int(rng.integers(0, len(days)))]
    return day.strftime("%Y-%m-%d")

def slugify(text: str, max_len: int = 40) -> str:
    # turn text into a clean identifier.
    s = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:max_len] if s else "COMPANY").upper()

# -----------------------------
# Request specs
# -----------------------------
@dataclasses.dataclass(frozen=True)
class ObsSpec:
    d: str
    k: int
    pr: int

@dataclasses.dataclass(frozen=True)
class RequestSpec:
    custom_id: str
    company: str
    obs: List[ObsSpec]

def load_companies(companies_csv: Path, company_col:str) -> List[str]:
    df = pd.read_csv(companies_csv)
    entities = list(df[company_col].unique())
    return entities


def build_request_specs(
    companies: Sequence[str],
    rows_per_company: int, # total firm-date observation per company
    obs_per_request: int, # how many firm-date observations per request
    seed: int,
) -> List[RequestSpec]:
    if obs_per_request < 1 or obs_per_request > 10:
        raise ValueError("obs_per_request must be in [1,10] (recommended 5–10).")
    rng = np.random.default_rng(seed)
    reqs: List[RequestSpec] = []

    for company in companies:
        slug = slugify(company)
        n = int(rows_per_company)
        n_reqs = int(math.ceil(n / obs_per_request))
        made = 0
        for ri in range(n_reqs):
            n_this = min(obs_per_request, n - made) # how many left
            obs: List[ObsSpec] = []
            for _ in range(n_this):
                d = random_2025_business_day(rng)
                k = sample_num_headlines(rng, max_k=25)
                pr = sample_num_press_releases(rng, k)
                obs.append(ObsSpec(d=d, k=k, pr=pr))
            made += n_this
            reqs.append(RequestSpec(custom_id=f"{slug}_{ri:05d}", company=company, obs=obs))
    return reqs

# -----------------------------
# Batch file writing: we build a list of API requests and write them into JSON file for processing
# -----------------------------
def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL with minified JSON to reduce file size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

def build_batch_line(
    req: RequestSpec,
    model: str,
    max_output_tokens: int,
    temperature: Optional[float],
    reasoning_effort: str,
    use_structured_outputs: bool,
    max_obs_per_request: int,
) -> Dict[str, Any]:
    user_payload = {"c": req.company, "obs": [dataclasses.asdict(o) for o in req.obs]}
    user_text = json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))

    body: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": reasoning_effort},
    }

    # GPT‑5: temperature supported only when reasoning.effort == "none"
    if temperature is not None and reasoning_effort == "none":
        body["temperature"] = float(temperature)

    if use_structured_outputs:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "syn_news",
                "strict": True,
                "schema": response_schema(max_obs_per_request),
            }
        }

    return {"custom_id": req.custom_id, "method": "POST", "url": "/v1/responses", "body": body}

def prepare_batch_files(
    reqs: Sequence[RequestSpec],
    out_dir: Path,
    model: str,
    max_output_tokens: int,
    temperature: Optional[float],
    reasoning_effort: str,
    use_structured_outputs: bool,
    max_obs_per_request: int,
    max_requests_per_file: int = 50_000,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest: custom_id -> company + obs specs (needed to map batch outputs back to firm-date rows)
    manifest_path = out_dir / "batch_manifest.jsonl"
    write_jsonl(manifest_path, ({"custom_id": r.custom_id, "company": r.company, "obs": [dataclasses.asdict(o) for o in r.obs]} for r in reqs))

    batch_paths: List[Path] = []
    for start in range(0, len(reqs), max_requests_per_file):
        chunk = reqs[start : start + max_requests_per_file]
        batch_path = out_dir / f"batch_input_{start:06d}.jsonl"
        write_jsonl(
            batch_path,
            (
                build_batch_line(
                    r,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    use_structured_outputs=use_structured_outputs,
                    max_obs_per_request=max_obs_per_request,
                )
                for r in chunk
            ),
        )

        # OpenAI Batch API input file size limit is 200MB
        size_bytes = batch_path.stat().st_size
        if size_bytes > 200 * 1024 * 1024:
            raise ValueError(
                f"Batch input file too large: {batch_path} is {size_bytes/1024/1024:.1f} MB (>200 MB). "
                "Increase obs_per_request, shorten prompts, or split into more files."
            )

        batch_paths.append(batch_path)

    return batch_paths

# -----------------------------
# Batch submit / poll / download: we submit the job, poll it constantly to see if it is done, then download it once it is done
# -----------------------------
def submit_batch_file(client: OpenAI, batch_input_path: Path, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uploaded = client.files.create(file=batch_input_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata or {},
    )
    return batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)

def poll_batch(client: OpenAI, batch_id: str) -> Dict[str, Any]:
    batch = client.batches.retrieve(batch_id)
    return batch.model_dump() if hasattr(batch, "model_dump") else dict(batch)

def download_file_content(client: OpenAI, file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id)
    out_path.write_bytes(content.read())

# -----------------------------
# Parse batch outputs
# -----------------------------
def extract_output_text(resp_body: Dict[str, Any]) -> Optional[str]:
    """
    Extract assistant output_text from a Responses API response body. Just parsing one response API
    """
    if not isinstance(resp_body, dict):
        return None
    if isinstance(resp_body.get("output_text"), str):
        return resp_body["output_text"]

    out = resp_body.get("output")
    if not isinstance(out, list):
        return None

    parts: List[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for p in content:
            if isinstance(p, dict) and p.get("type") == "output_text":
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
    return "".join(parts) if parts else None

def validate_observation(obs: Dict[str, Any], spec: ObsSpec) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if obs.get("d") != spec.d:
        errors.append(f"date_mismatch got={obs.get('d')} exp={spec.d}")

    h = obs.get("h")
    if not isinstance(h, list):
        return False, ["h_not_list"]

    if len(h) != spec.k:
        errors.append(f"k_mismatch got={len(h)} exp={spec.k}")

    pr_count = 0
    seen = set()
    for j, it in enumerate(h):
        if not isinstance(it, dict):
            errors.append(f"h{j}_not_obj")
            continue
        t, s, r, p = it.get("t"), it.get("s"), it.get("r"), it.get("p")
        if not isinstance(t, str) or not t.strip():
            errors.append(f"h{j}_t_bad")
        else:
            norm = re.sub(r"\s+", " ", t.strip().lower())
            if norm in seen:
                errors.append(f"h{j}_dup")
            seen.add(norm)

        if s not in (NEWS_SOURCES + PR_SOURCES):
            errors.append(f"h{j}_s_bad:{s}")

        if not isinstance(r, int) or not (51 <= r <= 100):
            errors.append(f"h{j}_r_bad:{r}")

        if p not in (0, 1):
            errors.append(f"h{j}_p_bad:{p}")
        else:
            pr_count += int(p)

        if p == 1 and s not in PR_SOURCES:
            errors.append(f"h{j}_prflag_nonprsrc:{s}")
        if p == 0 and s not in NEWS_SOURCES:
            errors.append(f"h{j}_nonprflag_prsrc:{s}")

    if pr_count != spec.pr:
        errors.append(f"pr_mismatch got={pr_count} exp={spec.pr}")

    z = obs.get("z")
    if not isinstance(z, (int, float)) or not (-2.0 <= float(z) <= 2.0):
        errors.append(f"z_bad:{z}")

    return (len(errors) == 0), errors

def enrich_with_label(
    company: str,
    obs: Dict[str, Any],
    alpha: float,
    p0: float,
    flip_rate: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    z = float(obs["z"])
    p_pos = sigmoid(logit(p0) + alpha * z)
    final_prob = (1.0 - flip_rate) * p_pos + flip_rate * p0
    y = int(rng.random() < final_prob)

    h = obs["h"]
    return {
        "company": company,
        "date": obs["d"],
        "overnight_sign": y,
        "final_prob": float(final_prob),
        "alpha": float(alpha),
        "p0": float(p0),
        "flip_rate": float(flip_rate),
        "z": float(z),
        "headlines": [x["t"] for x in h],
        "sources": [x["s"] for x in h],
        "relevance_scores": [int(x["r"]) for x in h],
        "is_press_release": [int(x["p"]) for x in h],
    }

def firmdate_row(en: Dict[str, Any]) -> Dict[str, Any]:
    # Normalize to match typical vendor-normalized feeds (real file is lower-cased)
    heads = [re.sub(r"\s+", " ", str(h)).strip().lower() for h in en["headlines"]]
    srcs = en["sources"]
    rel = en["relevance_scores"]
    pr = en["is_press_release"]

    return {
        "company": en["company"],
        "trade_date": en["date"],
        "overnight_sign": en["overnight_sign"],
        # keep the original field name for backward compatibility
        "n_headlines_generated": len(heads),
        # optional alias to match your real-data column naming convention
        "n_headlines_unique": len(heads),
        "n_sources": len(set(srcs)),
        "avg_relevance": float(np.mean(rel)) if rel else float("nan"),
        "pr_proportion": float(np.mean(pr)) if pr else 0.0,
        "headlines": json.dumps(heads, ensure_ascii=False),
        "sources": json.dumps(srcs, ensure_ascii=False),
    }

def parse_outputs(
    out_dir: Path,
    alpha: float,
    p0: float,
    flip_rate: float,
    seed: int,
) -> Tuple[Path, Path]:
    manifest_path = out_dir / "batch_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path} (run stage=prepare/submit first).")

    # custom_id -> (company, [ObsSpec...])
    manifest: Dict[str, Tuple[str, List[ObsSpec]]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            company = obj["company"]
            specs = [ObsSpec(**o) for o in obj["obs"]]
            manifest[cid] = (company, specs)

    output_files = sorted(out_dir.glob("batch_output_*.jsonl"))
    if not output_files:
        raise FileNotFoundError(f"No batch_output_*.jsonl in {out_dir} (run stage=download).")

    rng = np.random.default_rng(seed + 12345)

    enriched_path = out_dir / "synthetic_enriched.jsonl"
    csv_path = out_dir / "synthetic_firmdate.csv"
    bad_path = out_dir / "synthetic_bad_rows.jsonl"
    usage_path = out_dir / "token_usage_summary.json"

    # enriched_path = out_dir / "synthetic_enriched_v2.jsonl"
    # csv_path = out_dir / "synthetic_firmdate_v2.csv"
    # bad_path = out_dir / "synthetic_bad_rows_v2.jsonl"
    # usage_path = out_dir / "token_usage_summary_v2.json"

    enriched_path.write_text("", encoding="utf-8")
    bad_path.write_text("", encoding="utf-8")

    total_in = 0
    total_out = 0
    total_reasoning = 0

    csv_fields = [
        "company",
        "trade_date",
        "overnight_sign",
        "n_headlines_generated",
        "n_headlines_unique",
        "n_sources",
        "avg_relevance",
        "pr_proportion",
        "headlines",
        "sources",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()

        with enriched_path.open("a", encoding="utf-8") as ef, bad_path.open("a", encoding="utf-8") as bf:
            for ofile in output_files: # ususally just one
                with ofile.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        cid = row.get("custom_id")
                        if cid not in manifest:
                            bf.write(json.dumps({"kind": "unknown_custom_id", "row": row}, ensure_ascii=False) + "\n")
                            continue

                        company, specs = manifest[cid]

                        if row.get("error"):
                            bf.write(json.dumps({"kind": "api_error", "custom_id": cid, "error": row["error"]}, ensure_ascii=False) + "\n")
                            continue

                        resp = row.get("response", {})
                        body = resp.get("body", {})

                        # Token usage accounting (if present)
                        usage = body.get("usage") if isinstance(body, dict) else None
                        if isinstance(usage, dict):
                            it = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                            ot = usage.get("output_tokens") or usage.get("completion_tokens") or 0
                            total_in += int(it)
                            total_out += int(ot)
                            od = usage.get("output_tokens_details")
                            if isinstance(od, dict):
                                total_reasoning += int(od.get("reasoning_tokens") or 0)

                        out_text = extract_output_text(body)
                        if not out_text:
                            bf.write(json.dumps({"kind": "no_output_text", "custom_id": cid}, ensure_ascii=False) + "\n")
                            continue

                        try:
                            payload = json.loads(out_text)
                        except Exception as e:
                            bf.write(json.dumps({"kind": "json_parse_fail", "custom_id": cid, "err": str(e), "out_text": out_text[:300]}, ensure_ascii=False) + "\n")
                            continue

                        x = payload.get("x")
                        if not isinstance(x, list):
                            bf.write(json.dumps({"kind": "missing_x", "custom_id": cid, "payload": payload}, ensure_ascii=False) + "\n")
                            continue

                        # Validate and write each observation
                        m = min(len(x), len(specs))
                        if len(x) != len(specs):
                            bf.write(json.dumps({"kind": "x_len_mismatch", "custom_id": cid, "len_x": len(x), "len_specs": len(specs)}, ensure_ascii=False) + "\n")

                        for i in range(m):
                            obs = x[i]
                            spec = specs[i]
                            ok, errs = validate_observation(obs, spec)
                            if not ok:
                                bf.write(json.dumps({"kind": "invalid_obs", "custom_id": cid, "i": i, "errors": errs, "obs": obs}, ensure_ascii=False) + "\n")
                                continue

                            en = enrich_with_label(company, obs, alpha, p0, flip_rate, rng)
                            ef.write(json.dumps(en, ensure_ascii=False) + "\n")
                            writer.writerow(firmdate_row(en))

    summary = {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_reasoning_tokens": total_reasoning,
        "total_tokens": total_in + total_out,
    }
    usage_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return enriched_path, csv_path

# -----------------------------
# Main CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies_csv", type=Path, default=None, help="CSV with company names (top 40).")
    ap.add_argument("--company_col", type=str, default="entity_name_canon", help="Column name containing company names.")
    ap.add_argument("--rows_per_company", type=int, default=600, help="Firm-date observations per company.")
    ap.add_argument("--obs_per_request", type=int, default=10, help="Observations per API request")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    ap.add_argument("--model", type=str, default="gpt-5.2", help="Model name (one model per batch file).")
    ap.add_argument("--max_output_tokens", type=int, default=7500, help="Max output tokens per request.")
    ap.add_argument("--temperature", type=float, default=0.8, help="Temperature (only used if reasoning_effort=none).")
    ap.add_argument("--reasoning_effort", type=str, default="none", choices=["none", "low", "medium", "high"], help="Reasoning effort.")
    ap.add_argument("--no_structured_outputs", action="store_true", help="Disable JSON-schema structured outputs.")
    ap.add_argument("--seed", type=int, default=2026, help="Seed for sampling.")
    ap.add_argument("--alpha", type=float, default=1, help="Alpha in logit mapping from z to prob.")
    ap.add_argument("--p0", type=float, default=0.54, help="Baseline positive rate.")
    ap.add_argument("--flip_rate", type=float, default=0, help="Noise rate (shrink toward p0).")
    ap.add_argument("--stage", type=str, default="submit", choices=["prepare", "submit", "poll", "download", "parse"], help="Pipeline stage.")
    ap.add_argument("--poll_interval", type=int, default=30, help="Seconds between polls.")
    ap.add_argument("--max_requests_per_file", type=int, default=50000, help="Max requests per batch input file.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare request specs
    if args.stage in ("prepare", "submit"):
        if args.companies_csv is None:
            raise ValueError("--companies_csv is required for stage prepare/submit.")
        companies = load_companies(args.companies_csv, args.company_col)
        if not companies:
            raise ValueError("No companies loaded from companies_csv.")
        reqs = build_request_specs(companies, args.rows_per_company, args.obs_per_request, args.seed)

        batch_paths = prepare_batch_files(
            reqs=reqs,
            out_dir=out_dir,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            use_structured_outputs=(not args.no_structured_outputs),
            max_obs_per_request=args.obs_per_request,
            max_requests_per_file=args.max_requests_per_file,
        )
        print(f"[prepare] wrote {len(batch_paths)} batch_input_*.jsonl files and manifest to {out_dir}")

    client = OpenAI()

    # Submit
    if args.stage == "submit":
        batch_paths = sorted(out_dir.glob("batch_input_*.jsonl"))
        if not batch_paths:
            raise FileNotFoundError(f"No batch_input_*.jsonl found in {out_dir} (run stage=prepare first).")

        batches: List[Dict[str, Any]] = []
        for p in batch_paths:
            meta = {"description": "synthetic overnight headlines", "batch_input": p.name, "model": args.model}
            b = submit_batch_file(client, p, metadata=meta)
            batches.append(b)
            bid = b.get("id")
            (out_dir / f"batch_{bid}.json").write_text(json.dumps(b, indent=2), encoding="utf-8")
            print(f"[submit] created batch {bid} for {p.name}")

        (out_dir / "batches_index.json").write_text(json.dumps(batches, indent=2), encoding="utf-8")
        print(f"[submit] saved batches_index.json")

    # Poll
    if args.stage == "poll":
        idx = out_dir / "batches_index.json"
        if not idx.exists():
            raise FileNotFoundError(f"Missing {idx} (run stage=submit first).")
        batches = json.loads(idx.read_text(encoding="utf-8"))
        for b in batches:
            bid = b["id"]
            st = poll_batch(client, bid)
            (out_dir / f"batch_{bid}_status.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
            print(f"[poll] {bid}: {st.get('status')}")

    # Download
    if args.stage == "download":
        idx = out_dir / "batches_index.json"
        if not idx.exists():
            raise FileNotFoundError(f"Missing {idx} (run stage=submit first).")
        batches = json.loads(idx.read_text(encoding="utf-8"))

        for b in batches:
            bid = b["id"]
            st = poll_batch(client, bid)
            (out_dir / f"batch_{bid}_status.json").write_text(json.dumps(st, indent=2), encoding="utf-8")
            status = st.get("status")
            print(f"[download] {bid}: {status}")
            out_id = st.get("output_file_id")
            err_id = st.get("error_file_id")

            if out_id:
                out_path = out_dir / f"batch_output_{bid}.jsonl"
                if not out_path.exists():
                    download_file_content(client, out_id, out_path)
                    print(f"[download] saved {out_path.name}")
            if err_id:
                err_path = out_dir / f"batch_error_{bid}.jsonl"
                if not err_path.exists():
                    download_file_content(client, err_id, err_path)
                    print(f"[download] saved {err_path.name}")

    # Parse
    if args.stage == "parse":
        enriched_path, csv_path = parse_outputs(
            out_dir=out_dir,
            alpha=args.alpha,
            p0=args.p0,
            flip_rate=args.flip_rate,
            seed=args.seed
        )
        print(f"[parse] wrote {enriched_path}")
        print(f"[parse] wrote {csv_path}")
        print(f"[parse] bad rows: {out_dir / 'synthetic_bad_rows.jsonl'}")
        print(f"[parse] token summary: {out_dir / 'token_usage_summary.json'}")

if __name__ == "__main__":
    main()
