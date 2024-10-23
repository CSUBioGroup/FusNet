import argparse
import torch
from torch.utils.data import DataLoader
from enformer_pytorch import Enformer, GenomeIntervalDataset
import pandas as pd
from tqdm import tqdm

# 定义命令行参数解析
parser = argparse.ArgumentParser(description='Run Enformer model with specified data files.')
parser.add_argument('--bed_file', type=str, required=True, help='Path to the BED file.')
parser.add_argument('--fasta_file', type=str, required=True, help='Path to the FASTA file.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for data loading.')
parser.add_argument('--output_prefix', type=str, default='data/', help='Prefix for output files.')

args = parser.parse_args()

# 读取参数
bed_file = args.bed_file
fasta_file = args.fasta_file
model_path = args.model_path
batch_size = args.batch_size
output_prefix = args.output_prefix

use_gpu = torch.cuda.is_available()
print(torch.__version__)

# 加载数据集
ds = GenomeIntervalDataset(
    bed_file=bed_file,
    fasta_file=fasta_file,
    return_seq_indices=True,
    shift_augs=(-2, 2),
    context_length=196_608,
)

# 加载模型
model = Enformer.from_pretrained(model_path).cuda()

# 数据加载器
loader = DataLoader(
    dataset=ds,
    batch_size=batch_size,
)

# 输出文件准备
out_GM12878_CTCF = open(f'{output_prefix}GM12878_CTCF.bedGraph', 'a')
out_K562_YY1 = open(f'{output_prefix}K562_YY1.bedGraph', 'a')
out_K562_POLR2A = open(f'{output_prefix}K562_POLR2A.bedGraph', 'a')
out_GM12878_RAD21 = open(f'{output_prefix}GM12878_RAD21.bedGraph', 'a')

out_GM12878_CTCF.write('track type=bedGraph\n')
out_K562_YY1.write('track type=bedGraph\n')
out_K562_POLR2A.write('track type=bedGraph\n')
out_GM12878_RAD21.write('track type=bedGraph\n')

# 读取 BED 文件数据
data = pd.read_csv(bed_file, sep='\t', header=None)

# 进行模型推理
model.eval()
with torch.no_grad():
    epoch = 0
    for input_data in tqdm(loader, desc="Inference Progress", leave=False):
        input_data = input_data.cuda()
        pred = model(input_data, head='human')  # (batch, 896, 5313)

        # [687, 973, 979, 984]
        GM12878_CTCF = pred[:, :, 687].cpu().detach().numpy()
        K562_YY1 = pred[:, :, 973].cpu().detach().numpy()
        K562_POLR2A = pred[:, :, 979].cpu().detach().numpy()
        GM12878_RAD21 = pred[:, :, 984].cpu().detach().numpy()

        for i in range(pred.shape[0]):
            start = int(data.loc[epoch * batch_size + i, 1])
            end = int(data.loc[epoch * batch_size + i, 2])
            bins_num = (end - start) // 128

            for j in range(bins_num):
                if j == bins_num - 1:
                    out_GM12878_CTCF.write(
                        data.loc[i, 0] + '\t' + str(start + 128 * j) + '\t' + str(end) +
                        '\t' + str(GM12878_CTCF[i][j]) + '\n')
                    out_K562_YY1.write(
                        data.loc[i, 0] + '\t' + str(start + 128 * j) + '\t' + str(end) +
                        '\t' + str(K562_YY1[i][j]) + '\n')
                    out_K562_POLR2A.write(
                        data.loc[i, 0] + '\t' + str(start + 128 * j) + '\t' + str(end) +
                        '\t' + str(K562_POLR2A[i][j]) + '\n')
                    out_GM12878_RAD21.write(
                        data.loc[i, 0] + '\t' + str(start + 128 * j) + '\t' + str(end) +
                        '\t' + str(GM12878_RAD21[i][j]) + '\n')
                else:
                    out_GM12878_CTCF.write(
                        data.loc[i, 0] + '\t' + str(start + 128 * j) + '\t' + str(start + 128 * (j + 1)) +
                        '\t' + str(GM12878_CTCF[i][j]) + '\n')
                    out_K562_YY1.write(
                        data.loc[i, 0] + '\t' + str(start + 128 * j) + '\t' + str(start + 128 * (j + 1)) +
                        '\t' + str(K562_YY1[i][j]) + '\n')
                    out_K562_POLR2A.write(
                        data.loc[i, 0] + '\t' + str(start + 128 * j) + '\t' + str(start + 128 * (j + 1)) +
                        '\t' + str(K562_POLR2A[i][j]) + '\n')
                    out_GM12878_RAD21.write(
                        data.loc[i, 0] + '\t' + str(start + 128 * j) + '\t' + str(start + 128 * (j + 1)) +
                        '\t' + str(GM12878_RAD21[i][j]) + '\n')
        epoch += 1

# 关闭输出文件
out_GM12878_CTCF.close()
out_K562_YY1.close()
out_K562_POLR2A.close()
out_GM12878_RAD21.close()
