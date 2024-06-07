'''
Code for reproducing the generation of the STAPLE agreement.
'''
import os
import copy
import json
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from typing import List
from tqdm import tqdm
from collections import defaultdict

# CPV
RATERS = ['1', '2', '3']


def staple(images: List[sitk.Image]):
    stapler = sitk.STAPLEImageFilter()
    stapler.SetForegroundValue(1)
    
    output_image = sitk.BinaryThreshold(stapler.Execute(*images), lowerThreshold=0.5, upperThreshold=1.0, insideValue=1, outsideValue=0)
    
    sen, spec = stapler.GetSensitivity(), stapler.GetSpecificity()
    return {"staple": sitk.GetArrayFromImage(output_image), "sen": sen, "spec": spec}

def staple_iowa():
    global RATERS

    tgt_imgs = {r: sitk.ReadImage(os.path.join("data", f"longciu_{r}_tgt.nii.gz")) for r in RATERS}
    reference_img = tgt_imgs["1"]
    reference_array = sitk.GetArrayFromImage(reference_img)

    # GGO staple
    tgt_ggo_imgs = []
    for r in RATERS:
        tgt_ggo_img = (sitk.GetArrayFromImage(tgt_imgs[r]) == 1).astype(np.uint8)
        tgt_ggo_imgs.append(tgt_ggo_img)

    # CON staple
    tgt_con_imgs = []
    for r in RATERS:
        tgt_con_img = (sitk.GetArrayFromImage(tgt_imgs[r]) == 2).astype(np.uint8)
        tgt_con_imgs.append(tgt_con_img)

    # INF staple
    tgt_inf_imgs = []
    for r in RATERS:
        tgt_inf_img = (sitk.GetArrayFromImage(tgt_imgs[r]) > 0).astype(np.uint8)
        tgt_inf_imgs.append(tgt_inf_img)

    final_inf = []
    final_con = []
    final_ggo = []
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for i in tqdm(range(90)):
        ggo_output = staple([sitk.GetImageFromArray(x[i]) for x in tgt_ggo_imgs])
        con_output = staple([sitk.GetImageFromArray(x[i]) for x in tgt_con_imgs])
        inf_output = staple([sitk.GetImageFromArray(x[i]) for x in tgt_inf_imgs])
        
        final_ggo.append(ggo_output["staple"])
        for j in range(3):
            stats["ggo"]["sen"][f"rater{j}"]["values"].append(ggo_output["sen"][j])
            stats["ggo"]["spec"][f"rater{j}"]["values"].append(ggo_output["spec"][j])

        final_con.append(con_output["staple"])
        for j in range(3):
            stats["con"]["sen"][f"rater{j}"]["values"].append(con_output["sen"][j])
            stats["con"]["spec"][f"rater{j}"]["values"].append(con_output["spec"][j])

        final_inf.append(inf_output["staple"])
        for j in range(3):
            stats["inf"]["sen"][f"rater{j}"]["values"].append(inf_output["sen"][j])
            stats["inf"]["spec"][f"rater{j}"]["values"].append(inf_output["spec"][j])
    
    ggo_array = np.stack(final_ggo, axis=0)
    con_array = np.stack(final_con, axis=0)
    inf_array = np.stack(final_inf, axis=0)
    ggo = sitk.GetImageFromArray(ggo_array)
    con = sitk.GetImageFromArray(con_array)
    inf = sitk.GetImageFromArray(inf_array)
    ggo.CopyInformation(reference_img)
    con.CopyInformation(reference_img)
    inf.CopyInformation(reference_img)
    sitk.WriteImage(ggo, os.path.join("data", "longciu_STAPLE_ggo.nii.gz"))
    sitk.WriteImage(con, os.path.join("data", "longciu_STAPLE_con.nii.gz"))
    sitk.WriteImage(inf, os.path.join("data", "longciu_STAPLE_inf.nii.gz"))
    merged = np.zeros_like(reference_array, dtype=np.uint8)
    merged[inf_array == 1] = 1
    merged[con_array == 1] = 2
    merged = sitk.GetImageFromArray(merged)
    merged.CopyInformation(reference_img)
    sitk.WriteImage(merged, os.path.join("data", "longciu_STAPLE_tgt.nii.gz"))
    
    for struct, stat in copy.deepcopy(stats).items():
        for metric, raters in stat.items():
            for rater_name, values in raters.items():
                stats[struct][metric][rater_name]["mean"] = np.array(values["values"]).mean()
                stats[struct][metric][rater_name]["std"] = np.array(values["values"]).std()

    try:
        with open(os.path.join("data", "staple_pub_stats.json"), 'w') as staple_stats_file:
            json.dump(stats, staple_stats_file)
    except Exception as e:
        print(e)
        
if __name__ == "__main__":
    staple_iowa()
    