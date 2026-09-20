import pytest
from image_aesthetics.embeddings.feature_contract import *

def contract():
    return {"features": {name: ({"dimension": 768} if name in DIMENSIONS else {"dimension": 1}) | {"dtype":"float32","range":[0,1],"direction":"higher_is_better","model_revision":"r1","checkpoint_sha256":"0"*64,"preprocessing":{"image_size":224},"normalization":"unit"} for name in FEATURES}, "dtype": "float32", "preprocessing": {"image_size": 224}, "scalar_normalization": "[0,1]"}

def test_identity_deterministic_and_immutable(tmp_path):
    c=contract(); digest="a"*64; a=cache_identity(digest,c); assert a==cache_identity(digest,c)
    p=tmp_path/"contract.json"; assert write_contract(p,input_manifest_sha256=digest,contract=c)==a
    with pytest.raises((FileExistsError, ValueError)): write_contract(p,input_manifest_sha256="different",contract=c)

def test_reject_missing_unknown_cosine_and_absolute_path():
    c=contract(); del c["features"]["nima"]
    with pytest.raises(ValueError): validate_features(c["features"])
    c=contract(); c["features"]["cosine"]={"dimension":1}
    with pytest.raises(ValueError): validate_features(c["features"])
    with pytest.raises(ValueError): validate_relative_manifest([{"image_id":"x","source_path":"/tmp/x.jpg"}])
