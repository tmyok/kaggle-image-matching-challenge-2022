import hydra
import os
import pandas as pd
import time
import torch

from omegaconf import DictConfig

from exec_test_data import seed_everything
from exec_test_data import load_model_LoFTR, load_model_SE2LoFTR
from exec_test_data import load_model_DKM
from exec_test_data import load_model_SGMNet
from exec_test_data import preprocess, matching, postprocess
from validation import evaluate

from tools.draw_result import draw_matching

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    input_dir = os.path.join(cwd, "../input/image-matching-challenge-2022/train")
    csv_dir = os.path.join(cwd, "../input/imc2022-validation-csv")
    weight_dir = os.path.join(cwd, "../input/imc2022-pretrained-weights")
    output_dir = os.path.join(cwd, "../output")
    working_dir = os.path.join(cwd, "../output")

    covisibility_thr_min, covisibility_thr_max = cfg.covisibility_thr

    seed_everything(0)
    device = torch.device("cuda")
    ext = "jpg"

    #--------------------------------------------------------------------------------
    # parameters
    DKM_param = {
        "model": load_model_DKM(os.path.join(weight_dir, "dkm"), working_dir),
        "scale": cfg.DKM.scale,
        "sample_num": cfg.DKM.sample_num,
        "nms_ret_points": cfg.DKM.nms_ret_points,
        "mkpts_num_thr": cfg.DKM.mkpts_num_thr,
    }
    LoFTR_param = {
        "device": device,
        "model": load_model_LoFTR(os.path.join(weight_dir, "kornia-loftr", "loftr_outdoor.ckpt"), device),
        "img_size": cfg.LoFTR.img_size,
        "nms_ret_points": cfg.LoFTR.nms_ret_points,
        "conf_min_thr": cfg.LoFTR.conf_min_thr,
    }
    SE2LoFTR_param = {
        "device": device,
        "model": load_model_SE2LoFTR(os.path.join(weight_dir, "se2loftr", "4rot-big.ckpt"), device),
        "img_size": cfg.SE2LoFTR.img_size,
        "nms_ret_points": cfg.SE2LoFTR.nms_ret_points,
        "conf_min_thr": cfg.SE2LoFTR.conf_min_thr,
    }
    SGM_param = {
        "model": load_model_SGMNet(os.path.join(weight_dir, "sgmnet", "sgm", "root")),
        "mkpts_num_thr": cfg.SGMNet.mkpts_num_thr,
    }
    findFMat_param = {
        "input_num": cfg.findFMat.input_num,
        "ransacReprojThreshold": cfg.findFMat.ransacReprojThreshold,
        "confidence": cfg.findFMat.confidence,
        "maxIters": cfg.findFMat.maxIters,
    }

    params = {
        "LoFTR": LoFTR_param,
        "SE2LoFTR": SE2LoFTR_param,
        "DKM": DKM_param,
        "SGMNet": SGM_param,
        "findFMat": findFMat_param,
    }
    # parameters
    #--------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------
    # Load scene list
    df = pd.read_csv(os.path.join(input_dir, "scaling_factors.csv"))
    scene_list = df["scene"].unique()

    if cfg.scene != "all":
        scene_list = [cfg.scene]
    # Load scene list
    #--------------------------------------------------------------------------------

    total_time = 0
    sample_id_list_all = []
    fund_matrix_list_all = []
    for scene in scene_list:

        image_dir = os.path.join(input_dir, scene, "images")

        #--------------------------------------------------------------------------------
        # Cleaning data for validation
        #
        # https://www.kaggle.com/code/tmyok1984/imc2022-validation-code
        # https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328854#1812876

        # load train.csv
        df = pd.read_csv(os.path.join(csv_dir, f"{scene}.csv"))
        # remove low covisibility pairs
        df = df[(df["covisibility"] > covisibility_thr_min) & (df["covisibility"] < covisibility_thr_max)]
        # remove small images
        df = df[df["min_longest_edge"] > cfg.min_longest_edge]
        # remove image pairs separated in the depth direction
        df = df[df["dZ"] < cfg.dZ_thr]
        # remove images that are heavily tilted
        df = df[(df["image_1_rotation"] < cfg.rotation_thr) & (df["image_2_rotation"] < cfg.rotation_thr)]

        # random sample
        df = df.sample(n=min(len(df), cfg.max_num_pairs), random_state=42)

        # Cleaning data for validation
        #--------------------------------------------------------------------------------

        sample_ids = df["sample_id"].values
        image_1_ids = df["image_1_id"].values
        image_2_ids = df["image_2_id"].values

        scene_time = 0
        sample_id_list = []
        fund_matrix_list = []
        for sample_id, image_1_id, image_2_id in zip(sample_ids, image_1_ids, image_2_ids):

            start = time.time()
            args_1 = {
                "sample_id": sample_id,
                "image_dir": image_dir,
                "image_1_id": image_1_id,
                "image_2_id": image_2_id,
                "ext": ext,
            }
            args = {
                "args_1": args_1,
                "args_2": None,
            }
            result = preprocess(args, params) # load images
            result = matching(result, params) # make DKM ROI
            args = {
                "args_1": None,
                "args_2": result["args_1"],
            }
            result = preprocess(args, params) # SGMNet preprocess
            result = matching(result, params) # matching
            result = postprocess(result, params) # calc fMat
            result = result["args_2"]

            sample_id_list.append(sample_id)
            fund_matrix_list.append(result["F"])
            end = time.time()

            scene_time = scene_time + (end - start)
            if cfg.PRINT_INFO:
                print(f"{len(sample_id_list)} / {len(sample_ids)} : {sample_id}, elapsed time: {(end - start):.2f} sec.")

            if cfg.DEBUG:
                image_pair = sample_id.split(";")[2]
                image_1_id = image_pair.split("-")[0]
                image_2_id = image_pair.split("-")[1]
                draw_matching(
                    image_dir,
                    os.path.join(output_dir, f"{scene}-{image_1_id}-{image_2_id}_result.png"),
                    image_1_id, image_2_id, ext,
                    result["mkpts1"], result["mkpts2"], result["inliers"])

        if cfg.PRINT_INFO:
            print("Running time: ", scene_time, " s")
        total_time = total_time + scene_time

        if sample_id_list_all == []:
            sample_id_list_all = sample_id_list
            fund_matrix_list_all = fund_matrix_list
        else:
            sample_id_list_all.extend(sample_id_list)
            fund_matrix_list_all.extend(fund_matrix_list)

    if cfg.OUTPUT_RESULT:
        df = pd.DataFrame({"sample_id":sample_id_list_all, "fundamental_matrix":fund_matrix_list_all})
        outpur_csv = os.path.join(output_dir, f"train_pred_cov{covisibility_thr_min}-{covisibility_thr_max}_max_num_pairs{cfg.max_num_pairs}_{cfg.scene}.csv")
        df.to_csv(outpur_csv, index=False)

    # Evaluate submission
    maa, _, _ = evaluate(input_dir, sample_id_list_all, fund_matrix_list_all)
    print(f'mAA={maa:.05f} (n={len(sample_id_list_all)}), elapsed time: {total_time/60.0:.2f} min. -> {(total_time/len(sample_id_list_all)):.2f} sec/pair')

    return maa

if __name__ == "__main__":
    main()