import pandas as pd
from glob import glob

dic = dict()
dic["object"] = glob("../materials/objects/ShapeNetCore.v2/*/*/models/model_normalized.obj")
dic["paper_mesh"] = glob("../materials/paper_meshes/*.obj")
dic["doc"] = glob("../materials/docs/*.png")
dic["normal_map"] = glob("../materials/selected_normal_maps/*.png")
dic["envmap_indoor"] = glob("../materials/panoramas/indoor/*/*.png") + glob("../materials/panoramas/indoor/*/*.jpg")
dic["envmap_outdoor"] = glob("../materials/panoramas/outdoor/*/*.jpg")

for t in ["object", "paper_mesh", "doc", "normal_map", "envmap_indoor", "envmap_outdoor"]:
    df = pd.DataFrame()
    df["path"] = sorted(dic[t])
    df.to_csv(f"../materials/csv/{t}.csv", index=None)