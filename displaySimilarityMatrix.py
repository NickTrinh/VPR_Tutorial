import matplotlib.pyplot as plt
import numpy as np

def display_matches(S, img_db, img_q, thresh):
    """
    Display two images side by side and indicate whether they match.

    Args:
        M2 (np.ndarray): A boolean matrix indicating matches between database and query images.
        img_db (np.ndarray): The database image.
        img_q (np.ndarray): The query image.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display database image
    ax[0].imshow(img_db)
    ax[0].set_title('Database Image')
    ax[0].axis('off')

    # Display query image
    ax[1].imshow(img_q)
    ax[1].set_title('Query Image')
    ax[1].axis('off')

    # Check if the images match
    match = S[0, 0]
    match_text = 'match' if match > thresh else 'do not match'
    fig.suptitle(f'Images {match_text}', fontsize=16, color='green' if match > thresh else 'red')

    plt.show()