"""Runs ML Atlas practice problems under CPython with the same harness the browser uses.

Called by tests/problems.test.js with a JSON payload: {harness, problems, samples}.
Prints a JSON report to stdout.
"""
import contextlib
import io
import json
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")


def main(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    ns = {"__name__": "mla_harness"}
    exec(compile(payload["harness"], "<harness>", "exec"), ns)
    run_tests = ns["_mla_run_tests"]
    run_exec = ns["_mla_exec"]
    t0 = time.time()
    report = {"python": sys.version.split()[0], "problems": [], "samples": []}

    for p in payload["problems"]:
        spec = json.dumps({"prelude": p.get("prelude") or "", "solution": p["solution"], "tests": p["tests"]})
        item = {"id": p["id"], "total": len(p["tests"])}
        sol = json.loads(run_tests(p["solution"], spec))
        missing = _missing_module(sol)
        if missing:
            item["skipped"] = "needs " + missing
            report["problems"].append(item)
            continue
        item["solOk"] = sol["error"] is None and len(sol["tests"]) > 0 and all(t["passed"] for t in sol["tests"])
        if not item["solOk"]:
            item["solDetail"] = _describe(sol)
        st = json.loads(run_tests(p["starter"], spec))
        item["starterFails"] = st["error"] is not None or any(not t["passed"] for t in st["tests"])
        report["problems"].append(item)

    for s in payload["samples"]:
        item = {"id": s["id"]}
        try:
            compile(s["code"], s["id"], "exec")
            item["compiles"] = True
        except SyntaxError as e:
            item["compiles"] = False
            item["detail"] = "SyntaxError line %s: %s" % (e.lineno, e.msg)
        if item["compiles"] and s.get("run"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                res = json.loads(run_exec(s["code"]))
            err = res.get("error")
            if err and err.get("type") == "ModuleNotFoundError":
                item["skipped"] = err["message"]
            else:
                item["ran"] = err is None
                if err:
                    item["detail"] = err["message"] + (" | " + err["trace"] if err.get("trace") else "")
                out = buf.getvalue() + (chr(10) + res["value"] if res.get("value") else "")
                if s.get("expect"):
                    item["expectMissing"] = [e for e in s["expect"] if e not in out]
                if os.environ.get("MLA_SAMPLE_OUTPUT"):
                    item["output"] = out[:3000]
        report["samples"].append(item)

    report["seconds"] = round(time.time() - t0, 1)
    sys.stdout.write(json.dumps(report))


def _missing_module(result):
    err = result.get("error")
    if err and err.get("type") == "ModuleNotFoundError":
        return err["message"]
    for t in result.get("tests", []):
        e = t.get("error")
        if e and e.get("type") == "ModuleNotFoundError":
            return e["message"]
    return None


def _describe(result):
    if result.get("refError"):
        return result["refError"]
    if result.get("error"):
        e = result["error"]
        return e["message"] + (" | " + e["trace"] if e.get("trace") else "")
    for t in result["tests"]:
        if not t["passed"]:
            msg = "%s: got %s expected %s" % (t["name"], t.get("got"), t.get("expected"))
            if t.get("error"):
                msg += " | " + t["error"]["message"] + (" | " + t["error"]["trace"] if t["error"].get("trace") else "")
            return msg
    return "no tests"


if __name__ == "__main__":
    main(sys.argv[1])
