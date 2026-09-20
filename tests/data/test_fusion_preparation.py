from pathlib import Path
from PIL import Image
from image_aesthetics.data.fusion_preparation import group_exact, near_duplicate_candidates, assign_global_folds

def row(dataset, image, sha, score):
    return {"dataset_id": dataset, "image_id": image, "image_name": image, "record_id": f"{dataset}:{image}", "sha256": sha, "native_score": score}

def test_exact_groups_are_transitive_across_datasets():
    records=[row("a","1","same",1),row("b","2","same",2),row("c","3","same",3)]
    groups=group_exact(records)
    assert len(set(groups.values())) == 1

def test_dhash_multiband_recall_and_exact_exclusion(tmp_path: Path):
    # Deliberately differ in a high-order band while remaining close enough.
    a=Image.new("L",(64,64),128); b=Image.new("L",(64,64),128)
    for x in range(8): b.putpixel((x,0),255)
    a.save(tmp_path/"a.jpg"); b.save(tmp_path/"b.jpg")
    records=[row("a","a.jpg","sha-a",1),row("b","b.jpg","sha-b",2),row("c","c.jpg","sha-a",3)]
    roots={"a":tmp_path,"b":tmp_path,"c":tmp_path}
    candidates=near_duplicate_candidates(records,roots,threshold=8,workers=1)
    pairs={(x["left"],x["right"]) for x in candidates}
    assert ("a:a.jpg","b:b.jpg") in pairs or ("b:b.jpg","a:a.jpg") in pairs
    assert not any("c:c.jpg" in (x["left"],x["right"]) for x in candidates)

def test_fold_assignment_is_deterministic_and_populates_all_folds():
    records=[row("a",str(i),str(i),float(i)) for i in range(100)]
    groups={r["record_id"]:r["record_id"] for r in records}
    first=assign_global_folds(records,groups,folds=5,seed=7)
    second=assign_global_folds(records,groups,folds=5,seed=7)
    assert first==second and set(first.values())==set(range(5))
