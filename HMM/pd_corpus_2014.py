import os
import tarfile

# 解压缩
# 数据文件所在目录
base_dir = os.path.join(os.path.dirname(__file__),'nlp_data')

tar = tarfile.open(os.path.join(base_dir, 'people-2014.tar.gz'), 'r:gz', encoding='utf-8')

for tarinfo in tar:
    # 只读取文件内容
    if tarinfo.isreg():
        f = tar.extractfile(tarinfo)
        ctx = f.read()
        break

print(ctx.decode("utf-8"))