"""Two-stage refiner SFT data (45,610 rows): input = Dripper text, output = the refining teacher's tagged label
(<extract> | <refine>\n<text> | <delete>). The tagged file (input = page HTML, output = tagged label) is joined to the
Dripper SFT source (input = page HTML, output = Dripper text) by the HTML, so the student sees Dripper's text.
usage: build_t2t_set.py   (paths under $SFT_DIR, default $WORK_DIR/sft_data)"""
import json, os, shutil
D = os.environ.get("SFT_DIR", os.environ["WORK_DIR"] + "/sft_data")
def join(src, drip_files, out, vi_src, vi_out):
    drip = {}
    for f in drip_files:
        for l in open(D + "/" + f, encoding="utf-8"):
            d = json.loads(l); drip[d["input"]] = d["output"]
    n = miss = 0
    with open(D + "/" + out + ".tmp", "w", encoding="utf-8") as fo:
        for l in open(D + "/" + src, encoding="utf-8"):
            d = json.loads(l); n += 1; t = drip.get(d["input"])
            if t is None: miss += 1; t = ""
            fo.write(json.dumps({"input": t, "output": d["output"]}, ensure_ascii=False) + "\n")
    print(src, "rows", n, "missing dripper text", miss); assert miss == 0
    shutil.move(D + "/" + out + ".tmp", D + "/" + out)
    v = json.load(open(D + "/" + vi_src)); v["source"] = D + "/" + out; json.dump(v, open(D + "/" + vi_out, "w"))
join("two_stage/teacher_tagged.jsonl", ["two_stage/dripper_sft_source.jsonl"], "two_stage/two_stage_refiner_set.jsonl",
     "two_stage/teacher_tagged_val_idx.json", "two_stage/two_stage_refiner_set_val_idx.json")
