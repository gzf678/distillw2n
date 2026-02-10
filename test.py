import torch
import numpy as np
from collections import defaultdict

def print_ckpt_info(ckpt_path):
    """
    打印ckpt文件的所有键值信息，并检测NaN/Inf
    
    Args:
        ckpt_path: ckpt文件路径
    """
    # 1. 加载ckpt文件（map_location='cpu'避免GPU占用）
    print(f"=== 正在加载ckpt文件: {ckpt_path} ===")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"✅ 成功加载ckpt，顶层键: {list(ckpt.keys())}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 2. 定义检测函数
    def check_tensor(tensor, key_path):
        """检测单个张量的NaN/Inf情况"""
        if not isinstance(tensor, torch.Tensor):
            return {
                'type': type(tensor).__name__,
                'is_tensor': False,
                'has_nan': False,
                'has_inf': False,
                'info': str(tensor)[:100]  # 截断长字符串
            }
        
        # 张量统计信息
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total_elem = tensor.numel()
        
        # 关键统计（避免全部为NaN/Inf时报错）
        try:
            min_val = tensor.min().item() if nan_count < total_elem else float('nan')
            max_val = tensor.max().item() if nan_count < total_elem else float('nan')
            mean_val = tensor.mean().item() if nan_count < total_elem else float('nan')
        except:
            min_val = max_val = mean_val = float('nan')
        
        return {
            'type': type(tensor).__name__,
            'is_tensor': True,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'has_nan': nan_count > 0,
            'has_inf': inf_count > 0,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'total_elem': total_elem,
            'min': min_val,
            'max': max_val,
            'mean': mean_val
        }
    
    # 3. 递归遍历所有键值对
    def traverse_ckpt(data, parent_key="", results=None, nan_inf_keys=None):
        if results is None:
            results = {}
        if nan_inf_keys is None:
            nan_inf_keys = []
        
        if isinstance(data, dict):
            for k, v in data.items():
                current_key = f"{parent_key}.{k}" if parent_key else k
                traverse_ckpt(v, current_key, results, nan_inf_keys)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                current_key = f"{parent_key}[{i}]"
                traverse_ckpt(v, current_key, results, nan_inf_keys)
        else:
            # 检测当前值
            info = check_tensor(data, parent_key)
            results[parent_key] = info
            
            # 记录包含NaN/Inf的键
            if info.get('has_nan', False) or info.get('has_inf', False):
                nan_inf_keys.append(parent_key)
        
        return results, nan_inf_keys
    
    # 4. 执行遍历和检测
    print("\n=== 开始遍历ckpt内容并检测NaN/Inf ===")
    all_results, nan_inf_keys = traverse_ckpt(ckpt)
    
    # 5. 打印详细信息（可根据需要调整打印粒度）
    # 5.1 先打印包含NaN/Inf的键（重点关注）
    print(f"\n=== 包含NaN/Inf的键 ({len(nan_inf_keys)}个) ===")
    if nan_inf_keys:
        for key in nan_inf_keys:
            info = all_results[key]
            print(f"\n🔴 {key}")
            print(f"   - 类型: {info['type']}")
            if info['is_tensor']:
                print(f"   - 形状: {info['shape']}, 数据类型: {info['dtype']}")
                print(f"   - NaN数量: {info['nan_count']}/{info['total_elem']}")
                print(f"   - Inf数量: {info['inf_count']}/{info['total_elem']}")
                print(f"   - 统计: min={info['min']:.6f}, max={info['max']:.6f}, mean={info['mean']:.6f}")
            else:
                print(f"   - 内容: {info['info']}")
    else:
        print("🟢 未发现任何包含NaN/Inf的键")
    
    # 5.2 打印state_dict的所有键（模型权重，按模块分类）
    print(f"\n=== 模型权重（state_dict）完整列表 ===")
    state_dict = ckpt.get('state_dict', {})
    if state_dict:
        # 按模块分组
        module_groups = defaultdict(list)
        for key in state_dict.keys():
            # 拆分模块名（如 'encoder.layer1.weight' → 模块名 'encoder'）
            module_name = key.split('.')[0] if '.' in key else 'root'
            module_groups[module_name].append(key)
        
        for module, keys in module_groups.items():
            print(f"\n📦 模块: {module} ({len(keys)}个参数)")
            # 对每个键，简要打印是否包含NaN/Inf
            for key in keys:
                info = all_results[f"state_dict.{key}"]
                status = "🔴" if info['has_nan'] or info['has_inf'] else "🟢"
                print(f"   {status} {key}")
    else:
        print("⚠ 未找到state_dict")
    
    # 5.3 打印汇总统计
    print(f"\n=== 汇总统计 ===")
    total_keys = len(all_results)
    total_tensors = sum(1 for info in all_results.values() if info['is_tensor'])
    total_nan_inf = len(nan_inf_keys)
    
    print(f"📊 总键数: {total_keys}")
    print(f"📊 张量数量: {total_tensors}")
    print(f"📊 包含NaN/Inf的键数: {total_nan_inf}")
    print(f"📊 包含NaN/Inf的张量占比: {total_nan_inf/total_tensors*100:.2f}%" if total_tensors > 0 else "N/A")

# 执行检测
if __name__ == "__main__":
    # 替换为你的ckpt路径
    CKPT_PATH = "/mnt/workspace/guanzifan/distillw2n/experiments/s2uu2s/epoch.440-step.409942.ckpt"
    print_ckpt_info(CKPT_PATH)