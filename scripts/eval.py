#!/usr/bin/env python3
import argparse
import base64
import csv
import io
import json
import statistics
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


CLASSES = ("normal", "benign", "malignant")
BUSB_CLASSES = ("benign", "malignant")
PROMPT = """Choose exactly one category:
normal, benign, malignant."""


def images(root):
    for label in CLASSES:
        class_dir = root / label
        for path in sorted(class_dir.glob("*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and "_mask" not in path.stem:
                yield label, path, path.read_bytes()


def xlsx_rows(data):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        strings = []
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("a:si", ns):
            strings.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        for row in root.findall(".//a:row", ns):
            values = []
            for cell in row.findall("a:c", ns):
                value = cell.find("a:v", ns)
                if value is None:
                    values.append("")
                elif cell.attrib.get("t") == "s":
                    values.append(strings[int(value.text)])
                else:
                    values.append(value.text)
            yield values


def busb_images(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        rows = list(xlsx_rows(z.read("BUS/DatasetB.xlsx")))
        header = rows[0]
        image_col = header.index("Image")
        type_col = header.index("Type")
        for row in rows[1:]:
            if len(row) <= max(image_col, type_col):
                continue
            label = row[type_col].strip().lower()
            image_id = row[image_col].strip()
            name = f"BUS/original/{image_id}.png"
            if label in BUSB_CLASSES and name in z.namelist():
                yield label, Path(name), z.read(name)


def busbra_images(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        data = z.read("BUSBRA/bus_data.csv").decode("utf-8-sig").splitlines()
        for row in csv.DictReader(data):
            label = row["Pathology"].strip().lower()
            name = f"BUSBRA/Images/{row['ID'].strip()}.png"
            if label in BUSB_CLASSES and name in z.namelist():
                yield label, Path(name), z.read(name)


def ask_ollama(url, model, image_bytes, prompt, classes, timeout):
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = json.dumps({
        "model": model,
        "state": "image",
        "images": [image_b64],
        "questions": {
            "category": {
                "type": "choice",
                "instructions": prompt,
                "criteria": {label: None for label in classes},
            }
        },
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/systemone",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            answer = data["answers"]["category"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
    return answer.get("choice", "invalid"), json.dumps(answer, ensure_ascii=False)


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    index = round((len(values) - 1) * pct / 100)
    return values[index]


def print_report(rows, latencies, classes):
    labels = classes + ("invalid",)
    matrix = {true: {pred: 0 for pred in labels} for true in classes}
    for true, pred, *_ in rows:
        matrix[true][pred] += 1

    total = len(rows)
    correct = sum(matrix[label][label] for label in classes)
    print(f"total: {total}")
    print(f"accuracy: {correct / total:.4f}" if total else "accuracy: n/a")
    print()
    print("confusion_matrix rows=true cols=pred")
    print("true\\pred\t" + "\t".join(labels))
    for true in classes:
        print(true + "\t" + "\t".join(str(matrix[true][pred]) for pred in labels))
    print()
    print("per_class")
    for label in classes:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in classes if other != label)
        fn = sum(matrix[label][pred] for pred in labels if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{label}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print()
    print(
        "latency_sec "
        f"mean={statistics.mean(latencies):.3f} "
        f"p50={statistics.median(latencies):.3f} "
        f"p95={percentile(latencies, 95):.3f} "
        f"min={min(latencies):.3f} "
        f"max={max(latencies):.3f}"
    )


def logged_count(path):
    if not path or not path.exists():
        return 0
    with path.open(newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def rows_from_log(path):
    rows = []
    latencies = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["true"], row["pred"], float(row["latency_sec"]), Path(row["path"])))
            latencies.append(float(row["latency_sec"]))
    return rows, latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="BUSI class-directory root")
    parser.add_argument("--dataset", choices=("busi", "busb", "busbra"), default="busi")
    parser.add_argument("--busb-zip", type=Path, help="Dataset B BUS.zip")
    parser.add_argument("--busbra-zip", type=Path, help="BUSBRA.zip")
    parser.add_argument("--model", default="clef-flash:latest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.dataset == "busi" and not args.data:
        parser.error("--data is required for --dataset busi")
    if args.dataset == "busb" and not args.busb_zip:
        parser.error("--busb-zip is required for --dataset busb")
    if args.dataset == "busbra" and not args.busbra_zip:
        parser.error("--busbra-zip is required for --dataset busbra")
    if not args.prompt:
        args.prompt = PROMPT if args.dataset == "busi" else "Choose exactly one category:\nbenign, malignant."

    rows = []
    latencies = []
    classes = BUSB_CLASSES if args.dataset == "busb" else CLASSES
    if args.dataset == "busb":
        classes = BUSB_CLASSES
        samples = list(busb_images(args.busb_zip))
    elif args.dataset == "busbra":
        classes = BUSB_CLASSES
        samples = list(busbra_images(args.busbra_zip))
    else:
        classes = CLASSES
        samples = list(images(args.data))
    if args.limit:
        samples = samples[:args.limit]

    start_index = logged_count(args.log) if args.resume else 0
    log_file = args.log.open("a" if args.resume else "w", newline="") if args.log else None
    writer = csv.writer(log_file) if log_file else None
    if writer and not args.resume:
        writer.writerow(("index", "path", "true", "pred", "latency_sec", "raw"))
    try:
        for index, (true, path, image_bytes) in enumerate(samples[start_index:], start_index + 1):
            start = time.perf_counter()
            pred, raw = ask_ollama(args.ollama_url, args.model, image_bytes, args.prompt, classes, args.timeout)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            rows.append((true, pred, elapsed, path))
            if writer:
                writer.writerow((index, path, true, pred, f"{elapsed:.6f}", raw))
                log_file.flush()
            if not args.quiet:
                print(f"[{index}/{len(samples)}] true={true} pred={pred} latency={elapsed:.3f}s file={path.name} raw={raw!r}", flush=True)
    finally:
        if log_file:
            log_file.close()

    if args.resume and args.log:
        rows, latencies = rows_from_log(args.log)

    print()
    print_report(rows, latencies, classes)


if __name__ == "__main__":
    main()
