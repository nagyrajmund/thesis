import numpy as np
from scripts.integrated_gradients import relative_auc, describe_array

deletion_curves = np.load("../temp/deletion_curves.npy")
insertion_curves = np.load("../temp/insertion_curves.npy")
predicted_probs = np.load("../temp/prediction_confidences.npy")


for i in range(len(deletion_curves)):
    assert deletion_curves[i][0] == predicted_probs[i]
    if insertion_curves[i][-1] != predicted_probs[i]:
        print(insertion_curves[i][-1] - predicted_probs[i])


deletion_auc_scores_old = [relative_auc(del_curve, 1) for del_curve in deletion_curves]
deletion_auc_scores_new = [relative_auc(del_curve, del_curve[0]) for del_curve in deletion_curves]
print("Deletion: from {} to {}".format(describe_array(deletion_auc_scores_old), describe_array(deletion_auc_scores_new)))




insertion_auc_scores_old = [relative_auc(ins_curve, 1) for ins_curve in insertion_curves]
insertion_auc_scores_new = [relative_auc(ins_curve, ins_curve[-1]) for ins_curve in insertion_curves]
print("insertion: from {} to {}".format(describe_array(insertion_auc_scores_old), describe_array(insertion_auc_scores_new)))


black baseline, linear-latent interpolation,\
with signed attributions,\
black deletion, 3000 images, 30 steps,\
50 insertion bins, 50 deletion bins