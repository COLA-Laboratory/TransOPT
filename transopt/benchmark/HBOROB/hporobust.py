# Install robustbench if you haven't already
# !pip install robustbench

from robustbench.utils import load_model
from robustbench.data import load_cifar10
from robustbench.eval import benchmark

# Step 1: Load a pre-trained robust model from RobustBench
model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')

# Step 2: Load the CIFAR-10 test dataset
x_test, y_test = load_cifar10(n_examples=1000)

# Step 3: Evaluate the model's robustness
# We will use the AutoAttack suite to evaluate the model.
from robustbench.utils import clean_accuracy, AutoAttack

# Evaluate clean accuracy
clean_acc = clean_accuracy(model, x_test, y_test)
print(f'Clean accuracy: {clean_acc * 100:.2f}%')

# Step 4: Perform adversarial evaluation using AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255)
adv_acc = adversary.run_standard_evaluation(x_test, y_test, bs=128)

print(f'Robust accuracy against AutoAttack: {adv_acc * 100:.2f}%')