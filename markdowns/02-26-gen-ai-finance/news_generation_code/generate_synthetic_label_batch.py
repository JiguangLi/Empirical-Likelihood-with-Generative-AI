from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional until submit/poll/download
    OpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Source normalization (compact codes reduce batch file size / token use)
# ---------------------------------------------------------------------
SOURCE_CODE_MAP: Dict[str, str] = {
    "Barron's": "BRN",
    "Business Wire": "BW",
    "Canadian News Wire": "CNW",
    "Cision News": "Cision",
    "DGAP News": "DGAP",
    "DJ Global Press Release Wire": "DJPR",
    "Dow Jones Newswires": "DJN",
    "GlobeNewswire": "GNW",
    "HK Exchange News": "HKEX",
    "Hugin GlobeNewswire": "HGNW",
    "JSE News": "JSE",
    "LSE Regulatory News Service (RNS)": "RNS",
    "MarketWatch": "MW",
    "Oslo Bors News": "OSE",
    "PR Newswire": "PRN",
    "Wall Street Journal": "WSJ",
}

SOURCE_CODE_HELP = """
Source codes:
- More independent/editorial/newswire: DJN, WSJ, MW, BRN
- Press-release / distribution / exchange / regulatory style: PRN, BW, GNW, HGNW, Cision, CNW, DJPR, DGAP, HKEX, JSE, RNS, OSE
These weaker/distribution-style items still matter, but usually deserve less weight than independent reporting.
""".strip()

SYSTEM_PROMPT = f"""You are an annotator of overnight market-news tone for US equities.

Task
----
For each firm-date observation, read the supplied REAL headlines and aligned source codes.
Return n independent draws of a latent net overnight continous tone score z in [-2, 2].

Interpretation of z
-------------------
- +2.0 : very bullish / strong positive catalyst
- +1.0 : moderately positive
-  0.0 : mixed, neutral, or only weakly informative
- -1.0 : moderately negative
- -2.0 : very bearish / strong negative catalyst

Guidance
--------
1) Use only the supplied headlines and source codes.
2) You are not asked to infer the realized future return exactly; instead, score
   the news tone a plausible market participant might perceive overnight.
3) Many nights are mixed or weakly informative. Most draws should be near zero.
   Extreme values should be rare and reserved for clearly strong catalysts.
4) Administrative, exchange, filing, promotional, or routine press-release items
   are usually weaker evidence than independent reported news.
5) Analyst rating / price-target changes are moderate evidence.
6) Strong earnings/guidance surprises, major litigation/regulatory outcomes,
   financing stress, M&A, management shocks, outages, or clearly material product
   news can justify larger |z|.
7) Draws should vary modestly around your central judgment:
   - more dispersion when the evidence is mixed or ambiguous
   - tighter draws when the catalyst is clear
8) Do not output explanations.

{SOURCE_CODE_HELP}

INPUT JSON
----------
{{
  "obs": [
    {{
      "i": 0,
      "g": 0,
      "n": 5,
      "d": "YYYY-MM-DD",
      "p": [{{"h":"headline text","s":"DJN"}}, ...]
    }},
    ...
  ]
}}

OUTPUT JSON
-----------
{{
  "x": [
    {{
      "i": 0,
      "g": 0,
      "z": [0.1, -0.2, 0.0, ...]
    }},
    ...
  ]
}}

Return JSON only.
"""


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sanitize_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_source_code(s: Any) -> str:
    s = str(s)
    if s in SOURCE_CODE_MAP:
        return SOURCE_CODE_MAP[s]
    t = re.sub(r"[^A-Za-z0-9]+", "", s).upper()
    return t[:12] if t else "UNK"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def load_firm_date_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["headlines", "sources"]:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    return df




def get_openai_client():
    if OpenAI is None:
        raise ImportError(
            "The 'openai' package is required for stage submit / poll / download. "
            "Install it with: pip install openai"
        )
    return OpenAI()


def determine_p0(df: pd.DataFrame, p0_arg: str) -> float:
    if p0_arg.lower() == "auto":
        if "overnight_sign" not in df.columns:
            raise ValueError("p0='auto' requires an overnight_sign column in input_csv.")
        return float(df["overnight_sign"].astype(int).mean())
    return float(p0_arg)


# ---------------------------------------------------------------------
# Response schema (Structured Outputs)
# ---------------------------------------------------------------------
def response_schema(max_obs_per_request: int, max_draws_per_obs: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "x": {
                "type": "array",
                "minItems": 1,
                "maxItems": int(max_obs_per_request),
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "g": {"type": "integer", "minimum": 0},
                        "z": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": int(max_draws_per_obs),
                            "items": {
                                "type": "number",
                                "minimum": -2.0,
                                "maximum": 2.0,
                            },
                        },
                    },
                    "required": ["i", "g", "z"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["x"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------
# Request specs
# ---------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ObsChunkSpec:
    row_idx: int
    draw_chunk: int
    draw_start: int
    n_draws: int


@dataclasses.dataclass(frozen=True)
class RequestSpec:
    custom_id: str
    obs: List[ObsChunkSpec]


def build_request_specs(
    df: pd.DataFrame,
    draws_per_obs: int,
    draws_per_request: int,
    obs_per_request: int,
    seed: int,
    shuffle_chunks: bool,
) -> List[RequestSpec]:
    if draws_per_obs < 1:
        raise ValueError("draws_per_obs must be >= 1.")
    if draws_per_request < 1:
        raise ValueError("draws_per_request must be >= 1.")
    if obs_per_request < 1 or obs_per_request > 20:
        raise ValueError("obs_per_request must be in [1, 20].")

    chunks: List[ObsChunkSpec] = []
    for row_idx in df.index.tolist():
        chunk_id = 0
        for draw_start in range(0, draws_per_obs, draws_per_request):
            n_this = min(draws_per_request, draws_per_obs - draw_start)
            chunks.append(
                ObsChunkSpec(
                    row_idx=int(row_idx),
                    draw_chunk=int(chunk_id),
                    draw_start=int(draw_start),
                    n_draws=int(n_this),
                )
            )
            chunk_id += 1

    if shuffle_chunks:
        rng = np.random.default_rng(seed)
        rng.shuffle(chunks)

    reqs: List[RequestSpec] = []
    for ridx, start in enumerate(range(0, len(chunks), obs_per_request)):
        reqs.append(
            RequestSpec(
                custom_id=f"REQ_{ridx:07d}",
                obs=chunks[start : start + obs_per_request],
            )
        )
    return reqs


def build_obs_payload(row: pd.Series, spec: ObsChunkSpec) -> Dict[str, Any]:
    headlines = row["headlines"]
    sources = row["sources"]
    if not isinstance(headlines, list) or not isinstance(sources, list) or len(headlines) != len(sources):
        raise ValueError(
            f"Row {spec.row_idx} has invalid headlines/sources format: "
            f"headlines={type(headlines)}, sources={type(sources)}, len mismatch possible."
        )

    pairs = [
        {"h": sanitize_text(h), "s": normalize_source_code(s)}
        for h, s in zip(headlines, sources)
    ]
    return {
        "i": int(spec.row_idx),
        "g": int(spec.draw_chunk),
        "n": int(spec.n_draws),
        "d": str(row["trade_date"]) if "trade_date" in row.index else "",
        "p": pairs,
    }


def build_batch_line(
    req: RequestSpec,
    df: pd.DataFrame,
    model: str,
    max_output_tokens: int,
    temperature: Optional[float],
    reasoning_effort: str,
    use_structured_outputs: bool,
    max_obs_per_request: int,
    max_draws_per_obs: int,
) -> Dict[str, Any]:
    user_payload = {
        "obs": [build_obs_payload(df.loc[spec.row_idx], spec) for spec in req.obs]
    }
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

    if temperature is not None:
        body["temperature"] = float(temperature)

    if use_structured_outputs:
        body["text"] = {
            "format": {
                "type": "json_schema",
                "name": "synthetic_label_draws",
                "strict": True,
                "schema": response_schema(max_obs_per_request, max_draws_per_obs),
            }
        }

    return {
        "custom_id": req.custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def prepare_batch_files(
    df: pd.DataFrame,
    reqs: Sequence[RequestSpec],
    out_dir: Path,
    model: str,
    max_output_tokens: int,
    temperature: Optional[float],
    reasoning_effort: str,
    use_structured_outputs: bool,
    max_obs_per_request: int,
    max_draws_per_obs: int,
    max_requests_per_file: int = 50_000,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "batch_manifest.jsonl"
    write_jsonl(
        manifest_path,
        (
            {
                "custom_id": req.custom_id,
                "obs": [dataclasses.asdict(spec) for spec in req.obs],
            }
            for req in reqs
        ),
    )

    selected_rows_path = out_dir / "selected_row_indices.json"
    selected_rows = sorted({int(spec.row_idx) for req in reqs for spec in req.obs})
    selected_rows_path.write_text(json.dumps(selected_rows), encoding="utf-8")

    batch_paths: List[Path] = []
    for start in range(0, len(reqs), max_requests_per_file):
        chunk = reqs[start : start + max_requests_per_file]
        batch_path = out_dir / f"batch_input_{start:06d}.jsonl"
        write_jsonl(
            batch_path,
            (
                build_batch_line(
                    req=req,
                    df=df,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    use_structured_outputs=use_structured_outputs,
                    max_obs_per_request=max_obs_per_request,
                    max_draws_per_obs=max_draws_per_obs,
                )
                for req in chunk
            ),
        )

        size_bytes = batch_path.stat().st_size
        if size_bytes > 200 * 1024 * 1024:
            raise ValueError(
                f"Batch input file too large: {batch_path} is {size_bytes / 1024 / 1024:.1f} MB (>200 MB). "
                "Reduce obs_per_request, shorten prompts, or split into more files."
            )

        batch_paths.append(batch_path)

    return batch_paths


# ---------------------------------------------------------------------
# Batch submit / poll / download
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------
def extract_output_text(resp_body: Dict[str, Any]) -> Optional[str]:
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


def validate_output_item(item: Dict[str, Any], spec: ObsChunkSpec) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not isinstance(item, dict):
        return False, ["item_not_dict"]

    if item.get("i") != spec.row_idx:
        errors.append(f"row_idx_mismatch got={item.get('i')} exp={spec.row_idx}")
    if item.get("g") != spec.draw_chunk:
        errors.append(f"draw_chunk_mismatch got={item.get('g')} exp={spec.draw_chunk}")

    z = item.get("z")
    if not isinstance(z, list):
        errors.append("z_not_list")
        return False, errors

    if len(z) != spec.n_draws:
        errors.append(f"z_len_mismatch got={len(z)} exp={spec.n_draws}")

    for j, val in enumerate(z):
        if not isinstance(val, (int, float)):
            errors.append(f"z{j}_not_number")
            continue
        if not (-2.0 <= float(val) <= 2.0):
            errors.append(f"z{j}_out_of_range:{val}")

    return (len(errors) == 0), errors


def z_to_prob(z: float, label_alpha: float, p0: float, flip_rate: float) -> float:
    p_pos = sigmoid(logit(p0) + float(label_alpha) * float(z))
    final_prob = (1.0 - float(flip_rate)) * p_pos + float(flip_rate) * p0
    return float(final_prob)


def make_wide_output(
    df: pd.DataFrame,
    row_indices: List[int],
    z_mat: np.ndarray,
    y_mat: np.ndarray,
) -> pd.DataFrame:
    out = df.loc[row_indices].copy()

    n_draws = int(z_mat.shape[1])
    for j in range(n_draws):
        out[f"synthetic_z_{j + 1:03d}"] = z_mat[:, j]
    for j in range(n_draws):
        ser = pd.Series(y_mat[:, j], index=out.index)
        if np.isnan(ser).any():
            out[f"synthetic_overnight_sign_{j + 1:03d}"] = ser.astype("Int64")
        else:
            out[f"synthetic_overnight_sign_{j + 1:03d}"] = ser.astype(int)

    out["n_synthetic_draws_success"] = np.sum(~np.isnan(z_mat), axis=1).astype(int)
    out["synthetic_z_mean"] = np.nanmean(z_mat, axis=1)
    out["synthetic_z_sd"] = np.nanstd(z_mat, axis=1, ddof=1)
    out["synthetic_sign_mean"] = np.nanmean(y_mat, axis=1)
    return out


def parse_outputs(
    out_dir: Path,
    input_csv: Path,
    draws_per_obs: int,
    label_alpha: float,
    p0: float,
    flip_rate: float,
    seed: int,
) -> Tuple[Path, Path]:
    manifest_path = out_dir / "batch_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path} (run stage=prepare/submit first).")

    df = load_firm_date_csv(input_csv)

    selected_rows_path = out_dir / "selected_row_indices.json"
    if selected_rows_path.exists():
        row_indices = json.loads(selected_rows_path.read_text(encoding="utf-8"))
    else:
        row_indices = sorted(df.index.tolist())

    manifest: Dict[str, List[ObsChunkSpec]] = {}
    max_seen_draw = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            specs = [ObsChunkSpec(**o) for o in obj["obs"]]
            manifest[obj["custom_id"]] = specs
            for spec in specs:
                max_seen_draw = max(max_seen_draw, spec.draw_start + spec.n_draws)

    if max_seen_draw != draws_per_obs:
        # Use the manifest as source of truth if they differ.
        draws_per_obs = int(max_seen_draw)

    row_indices = sorted(int(i) for i in row_indices)
    rowpos = {row_idx: pos for pos, row_idx in enumerate(row_indices)}
    z_mat = np.full((len(row_indices), draws_per_obs), np.nan, dtype=float)
    y_mat = np.full((len(row_indices), draws_per_obs), np.nan, dtype=float)
    prob_mat = np.full((len(row_indices), draws_per_obs), np.nan, dtype=float)

    output_files = sorted(out_dir.glob("batch_output_*.jsonl"))
    if not output_files:
        raise FileNotFoundError(f"No batch_output_*.jsonl found in {out_dir} (run stage=download first).")

    bad_path = out_dir / "synthetic_label_bad_rows.jsonl"
    bad_path.write_text("", encoding="utf-8")

    draws_long_path = out_dir / "synthetic_label_draws.jsonl"
    draws_long_path.write_text("", encoding="utf-8")

    usage_path = out_dir / "token_usage_summary.json"
    wide_csv_path = out_dir / "synthetic_label_wide.csv"
    summary_csv_path = out_dir / "synthetic_label_summary.csv"

    rng = np.random.default_rng(int(seed) + 1729)
    total_in = 0
    total_out = 0
    total_reasoning = 0

    with bad_path.open("a", encoding="utf-8") as bf, draws_long_path.open("a", encoding="utf-8") as drawf:
        for ofile in output_files:
            with ofile.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cid = row.get("custom_id")

                    if cid not in manifest:
                        bf.write(json.dumps({"kind": "unknown_custom_id", "row": row}, ensure_ascii=False) + "\n")
                        continue

                    specs = manifest[cid]

                    if row.get("error"):
                        bf.write(json.dumps({"kind": "api_error", "custom_id": cid, "error": row["error"]}, ensure_ascii=False) + "\n")
                        continue

                    resp = row.get("response", {})
                    body = resp.get("body", {})

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
                        bf.write(
                            json.dumps(
                                {
                                    "kind": "json_parse_fail",
                                    "custom_id": cid,
                                    "err": str(e),
                                    "out_text": out_text[:500],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    x = payload.get("x")
                    if not isinstance(x, list):
                        bf.write(json.dumps({"kind": "missing_x", "custom_id": cid, "payload": payload}, ensure_ascii=False) + "\n")
                        continue

                    if len(x) != len(specs):
                        bf.write(
                            json.dumps(
                                {
                                    "kind": "x_len_mismatch",
                                    "custom_id": cid,
                                    "len_x": len(x),
                                    "len_specs": len(specs),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    m = min(len(x), len(specs))
                    for i in range(m):
                        item = x[i]
                        spec = specs[i]
                        ok, errs = validate_output_item(item, spec)
                        if not ok:
                            bf.write(
                                json.dumps(
                                    {
                                        "kind": "invalid_item",
                                        "custom_id": cid,
                                        "i": i,
                                        "errors": errs,
                                        "item": item,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            continue

                        if spec.row_idx not in rowpos:
                            bf.write(
                                json.dumps(
                                    {"kind": "row_idx_not_selected", "custom_id": cid, "row_idx": spec.row_idx},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            continue

                        rpos = rowpos[spec.row_idx]
                        z_list = [float(v) for v in item["z"]]

                        for local_idx, z in enumerate(z_list):
                            draw_idx = spec.draw_start + local_idx
                            if not np.isnan(z_mat[rpos, draw_idx]):
                                bf.write(
                                    json.dumps(
                                        {
                                            "kind": "duplicate_draw",
                                            "custom_id": cid,
                                            "row_idx": spec.row_idx,
                                            "draw_idx": draw_idx,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                continue

                            prob = z_to_prob(z, label_alpha=label_alpha, p0=p0, flip_rate=flip_rate)
                            y = int(rng.random() < prob)

                            z_mat[rpos, draw_idx] = z
                            prob_mat[rpos, draw_idx] = prob
                            y_mat[rpos, draw_idx] = y

                            drawf.write(
                                json.dumps(
                                    {
                                        "row_idx": int(spec.row_idx),
                                        "draw_idx": int(draw_idx),
                                        "draw_chunk": int(spec.draw_chunk),
                                        "synthetic_z": float(z),
                                        "synthetic_prob": float(prob),
                                        "synthetic_overnight_sign": int(y),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

    usage_summary = {
        "total_input_tokens": int(total_in),
        "total_output_tokens": int(total_out),
        "total_reasoning_tokens": int(total_reasoning),
        "total_tokens": int(total_in + total_out),
    }
    usage_path.write_text(json.dumps(usage_summary, indent=2), encoding="utf-8")

    wide_df = make_wide_output(df=df, row_indices=row_indices, z_mat=z_mat, y_mat=y_mat)
    wide_df.to_csv(wide_csv_path, index=False)

    summary_df = wide_df[["rp_entity_id", "trade_date", "n_synthetic_draws_success", "synthetic_z_mean", "synthetic_z_sd", "synthetic_sign_mean"]].copy()
    summary_df.to_csv(summary_csv_path, index=False)

    return wide_csv_path, summary_csv_path


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Generate synthetic label draws from real firm-date headline sets via OpenAI Batch API")
    ap.add_argument("--input_csv", type=Path, default=None, help="Firm-date input CSV (e.g., firm_date_train_r50.csv).")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    ap.add_argument("--model", type=str, default="gpt-5.2", help="Responses API model.")
    ap.add_argument("--max_rows", type=int, default=None, help="Optional row cap for debugging.")
    ap.add_argument("--draws_per_obs", type=int, default=200, help="Total z draws desired per input row.")
    ap.add_argument("--draws_per_request", type=int, default=10, help="How many z draws to request per observation, per API call.")
    ap.add_argument("--obs_per_request", type=int, default=5, help="How many observations to bundle into one request.")
    ap.add_argument("--max_output_tokens", type=int, default=1200, help="Max output tokens per request.")
    ap.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    ap.add_argument(
        "--reasoning_effort",
        type=str,
        default="none",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Responses API reasoning effort.",
    )
    ap.add_argument("--no_structured_outputs", action="store_true", help="Disable JSON-schema structured outputs.")
    ap.add_argument("--seed", type=int, default=2026, help="Seed used for request ordering / label draws.")
    ap.add_argument("--shuffle_chunks", action="store_true", help="Shuffle observation chunks before grouping into requests.")
    ap.add_argument("--label_alpha", type=float, default=1.0, help="Slope in the z -> probability mapping.")
    ap.add_argument("--p0", type=str, default="auto", help="Baseline positive rate; use 'auto' to infer from input overnight_sign.")
    ap.add_argument("--flip_rate", type=float, default=0.0, help="Noise rate that shrinks probabilities toward p0.")
    ap.add_argument("--stage", type=str, default="submit", choices=["prepare", "submit", "poll", "download", "parse"], help="Pipeline stage.")
    ap.add_argument("--max_requests_per_file", type=int, default=50_000, help="Max requests per batch input file.")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("prepare", "submit", "parse") and args.input_csv is None:
        raise ValueError("--input_csv is required for stage prepare / submit / parse.")

    if args.stage in ("prepare", "submit"):
        df = load_firm_date_csv(args.input_csv)
        if args.max_rows is not None:
            df = df.iloc[: int(args.max_rows)].copy()

        reqs = build_request_specs(
            df=df,
            draws_per_obs=int(args.draws_per_obs),
            draws_per_request=int(args.draws_per_request),
            obs_per_request=int(args.obs_per_request),
            seed=int(args.seed),
            shuffle_chunks=bool(args.shuffle_chunks),
        )

        batch_paths = prepare_batch_files(
            df=df,
            reqs=reqs,
            out_dir=out_dir,
            model=args.model,
            max_output_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature) if args.temperature is not None else None,
            reasoning_effort=args.reasoning_effort,
            use_structured_outputs=(not args.no_structured_outputs),
            max_obs_per_request=int(args.obs_per_request),
            max_draws_per_obs=int(args.draws_per_request),
            max_requests_per_file=int(args.max_requests_per_file),
        )
        print(f"[prepare] wrote {len(batch_paths)} batch_input_*.jsonl file(s) and manifest to {out_dir}")

    client = get_openai_client() if args.stage in {"submit", "poll", "download"} else None

    if args.stage == "submit":
        batch_paths = sorted(out_dir.glob("batch_input_*.jsonl"))
        if not batch_paths:
            raise FileNotFoundError(f"No batch_input_*.jsonl found in {out_dir} (run stage=prepare first).")

        batches: List[Dict[str, Any]] = []
        for p in batch_paths:
            meta = {
                "description": "synthetic label draws from real headline sets",
                "batch_input": p.name,
                "model": args.model,
            }
            b = submit_batch_file(client, p, metadata=meta)
            batches.append(b)
            bid = b.get("id")
            (out_dir / f"batch_{bid}.json").write_text(json.dumps(b, indent=2), encoding="utf-8")
            print(f"[submit] created batch {bid} for {p.name}")

        (out_dir / "batches_index.json").write_text(json.dumps(batches, indent=2), encoding="utf-8")
        print("[submit] saved batches_index.json")

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

    if args.stage == "parse":
        df_for_p0 = load_firm_date_csv(args.input_csv)
        if args.max_rows is not None:
            df_for_p0 = df_for_p0.iloc[: int(args.max_rows)].copy()
        p0 = determine_p0(df_for_p0, args.p0)

        wide_csv_path, summary_csv_path = parse_outputs(
            out_dir=out_dir,
            input_csv=args.input_csv,
            draws_per_obs=int(args.draws_per_obs),
            label_alpha=float(args.label_alpha),
            p0=float(p0),
            flip_rate=float(args.flip_rate),
            seed=int(args.seed),
        )
        print(f"[parse] p0 used = {p0:.6f}")
        print(f"[parse] wrote {wide_csv_path}")
        print(f"[parse] wrote {summary_csv_path}")
        print(f"[parse] bad rows: {out_dir / 'synthetic_label_bad_rows.jsonl'}")
        print(f"[parse] token summary: {out_dir / 'token_usage_summary.json'}")


if __name__ == "__main__":
    main()
