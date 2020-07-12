import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def variation_ratio(output):
    output = output.cpu()
    output_softmax = F.softmax(output, dim=1)
    n_samples = output.shape[2]
    c_star = torch.argmax(output_softmax, dim=1)
    count = torch.zeros(output.shape[0], 2)
    for i in range(count.shape[0]):
        c = np.bincount(c_star[i, :])
        count[i, 0] = torch.argmax(torch.from_numpy(c))
        count[i, 1] = c[int(count[i,0])]

    fx = count[:, 1]
    variation_ratio = 1 - fx / n_samples
    return  variation_ratio

def pridiction_entropy(output):
    output = output.cpu()
    output_softmax = F.softmax(output, dim=1)
    prob = torch.mean(output_softmax, dim=2)
    log_prob = torch.log(prob)
    pred_entropy = torch.sum((- prob * log_prob), dim=1)
    return pred_entropy

def mutual_information(output):
    output = output.cpu()
    pred_entropy = pridiction_entropy(output)
    output_softmax = F.softmax(output, dim=1)
    prob = output_softmax
    log_prob = torch.log(prob)
    MI = pred_entropy +  torch.mean(torch.sum((- prob * log_prob), dim=1), dim=1)
    return MI

def test_uncertainities(Outputs, targets, num_classes, logger, save_path, plot=True):
    Vr = variation_ratio(Outputs)
    Pe = pridiction_entropy(Outputs)
    Mi = mutual_information(Outputs)

    Outputs = F.log_softmax(Outputs, dim=1).cpu()
    targets = targets.cpu()

    predicted_for_images = 0
    correct_predictions = 0

    for i in range(len(targets)):

        if (plot):
            print("Real: ", targets[i])
            fig, axs = plt.subplots(1, num_classes, sharey=True, figsize=(20, 2))

        all_digits_prob = []

        highted_something = False

        for j in range(num_classes):

            highlight = False

            histo = []
            histo_exp = []

            for z in range(Outputs.shape[2]):
                histo.append(Outputs[i][j][z])
                histo_exp.append(np.exp(Outputs[i][j][z]))

            prob = np.percentile(histo_exp, 100 // 2)  # sampling median probability

            if (prob > 0.2):  # select if network thinks this sample is 20% chance of this being a label
                highlight = True  # possibly an answer

            all_digits_prob.append(prob)

            if (plot):
                N, bins, patches = axs[j].hist(histo, bins=8, color="lightgray", lw=0,
                                               weights=np.ones(len(histo)) / len(histo), density=False)
                axs[j].set_title("Class" + str(j + 1) + " (prob," + str(round(prob, 5)) + ")")
                axs[j].set_xlabel("Log_softmax")
                axs[j].set_ylabel("Percent")

            if (highlight):

                highted_something = True

                if (plot):

                    # We'll color code by height, but you could use any scalar
                    fracs = N / N.max()

                    # we need to normalize the data to 0..1 for the full range of the colormap
                    from matplotlib import colors
                    norm = colors.Normalize(fracs.min(), fracs.max())

                    # Now, we'll loop through our objects and set the color of each accordingly
                    for thisfrac, thispatch in zip(fracs, patches):
                        color = plt.cm.viridis(norm(thisfrac))
                        thispatch.set_facecolor(color)

        if (plot):
            plt.show()

        plt.savefig(save_path + '/' + str(i + 1) + '.png')

        logger.info( str(i+1))
        logger.info("Variation ratio: {:.3f}".format(Vr[i]))
        logger.info("Predictive Entropy: {:.3f}".format(Pe[i]))
        logger.info("Mutual Information: {:.3f}".format(Mi[i]))

        predicted = np.argmax(all_digits_prob)

        print()
        if (highted_something):
            predicted_for_images += 1
            if (targets[i] == predicted):
                if (plot):
                    logger.info("Correct")
                correct_predictions += 1.0
            else:
                if (plot):
                    logger.info("Incorrect :()")
        else:
            if (plot):
                logger.info("Undecided.")

        logger.info("\n")

        # if (plot):
        #     imshow(images[i].squeeze())

    if (plot):
        logger.info("Summary")
        logger.info("Total images: {}".format(len(targets)))
        logger.info("Predicted for: {}".format(predicted_for_images))
        logger.info("Accuracy when predicted: {}".format(correct_predictions / predicted_for_images))
