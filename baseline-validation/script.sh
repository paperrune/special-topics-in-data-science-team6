DEVICE=cuda:1
METHODS="msp energy mahalanobis vim vos npos"

# CIFAR10
DATASET=cifar10
EPOCHS=10
for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/pretrained
done

for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --no-pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/no-pretrained
done

# CIFAR100
DATASET=cifar100
EPOCHS=20
for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/pretrained
done

for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --no-pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/no-pretrained
done

# CUB200
DATASET=cub200
EPOCHS=30
for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/pretrained
done

for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --no-pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/no-pretrained
done

# Stanford Cars
DATASET=stanfordcars
EPOCHS=30
for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/pretrained
done

for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --no-pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/no-pretrained
done

# Oxford Pets
DATASET=oxfordpets
EPOCHS=20
for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/pretrained
done

for METHOD in $METHODS; do
    python3 main.py --dataset $DATASET --method $METHOD --epochs $EPOCHS --no-pretrained --device $DEVICE --output-dir saved/${DATASET}/${METHOD}/no-pretrained
done
