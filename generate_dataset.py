import glob

# train=glob.glob(r"M:\pycharmProjects\泡面\train\*.*g")
val=glob.glob(r"M:\pycharmProjects\泡面\val\*.*g")
# with open(r"M:\pycharmProjects\泡面\train.txt", "w") as op:
#     for i in train:
#         op.writelines(i+"\n")
#     op.close()
with open(r"M:\pycharmProjects\泡面\val.txt", "w") as op:
    for i in val:
        op.writelines(i+"\n")
    op.close()