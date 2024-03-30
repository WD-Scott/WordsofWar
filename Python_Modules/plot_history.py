import matplotlib.pyplot as plt

def plot_training_history(acc, val_acc, loss, val_loss, epochs_range):
    """
    Plots training and validation accuracy along with training and validation loss over epochs.

    Parameters:
    acc (numpy.ndarray): Training accuracy values.
    val_acc (numpy.ndarray): Validation accuracy values.
    loss (numpy.ndarray): Training loss values.
    val_loss (numpy.ndarray): Validation loss values.
    epochs_range (numpy.ndarray): Array of epoch values.

    Returns:
    None: This function does not return anything, it just plots the data.
    """
    plt.figure(figsize=(12, 5))
    plt.style.use('fivethirtyeight')
    
    # Subplot 1: Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training', color='#232D4B')
    plt.xlabel('Epoch', weight='bold', size=15)
    plt.ylabel('Accuracy', weight='bold', size=15)
    plt.plot(epochs_range, val_acc, label='Validation', color='#E57200')
    legend = plt.legend(loc='lower right')
    plt.setp(legend.texts, weight='bold', size=13)  # Set legend text to bold
    legend.get_frame().set_edgecolor('black')  # Set legend outline color to black
    plt.title('Training and Validation Accuracy', weight='bold', size=17)

    # Subplot 2: Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training', color='#232D4B')
    plt.xlabel('Epoch', weight='bold', size=15)
    plt.ylabel('Loss', weight='bold', size=15)
    plt.plot(epochs_range, val_loss, label='Validation', color='#E57200')
    legend = plt.legend(loc='upper right')
    plt.setp(legend.texts, weight='bold', size=13)  # Set legend text to bold
    legend.get_frame().set_edgecolor('black')  # Set legend outline color to black
    plt.title('Training and Validation Loss', weight='bold', size=17)

    plt.show()

# Example usage:
# plot_training_history(acc2, val_acc2, loss2, val_loss2, epochs_range)
