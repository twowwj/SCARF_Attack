import yaml
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os

@dataclass
class PoisonConfig:
    """数据毒化配置类"""
    # 基本毒化参数
    poison_ratio: float = 0.1
    trigger_path: Optional[str] = None
    trigger_position: str = 'bottom_right'
    trigger_size: Tuple[int, int] = (100, 100)
    trigger_type: str = 'image'  # 'image' or 'synthetic'
    synthetic_trigger_type: str = 'checkerboard'  # 合成触发器类型
    
    # 攻击目标参数
    attack_bbox_min: str = '-1,-1,-1'
    attack_bbox_max: str = '1,1,1'
    mu_vector: str = '0,0,0'
    
    # 损失权重参数
    beta: float = 1.0  # 攻击损失权重
    lambda_collapse: float = 1.0  # 坍塌损失权重
    
    # 训练参数
    iterations: int = 30000
    learning_rate: float = 0.001
    
    # 评估参数
    test_cameras: int = 100
    metrics: List[str] = field(default_factory=lambda: ['psnr', 'ssim', 'lpips'])
    
    def validate(self):
        """验证配置参数"""
        # 验证毒化比例
        if not 0.0 <= self.poison_ratio <= 1.0:
            raise ValueError(f"poison_ratio must be between 0.0 and 1.0, got {self.poison_ratio}")
        
        # 验证触发器类型和路径
        if self.trigger_type == 'image' and not self.trigger_path:
            raise ValueError("trigger_path must be provided when trigger_type is 'image'")
        
        if self.trigger_path and not os.path.exists(self.trigger_path):
            raise ValueError(f"Trigger file not found: {self.trigger_path}")
        
        # 验证触发器位置
        valid_positions = ['bottom_right', 'top_left', 'center', 'top_right', 'bottom_left']
        if self.trigger_position not in valid_positions:
            raise ValueError(f"trigger_position must be one of {valid_positions}")
        
        # 验证触发器尺寸
        if self.trigger_size[0] <= 0 or self.trigger_size[1] <= 0:
            raise ValueError(f"trigger_size must be positive, got {self.trigger_size}")
        
        # 验证损失权重
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        
        if self.lambda_collapse < 0:
            raise ValueError(f"lambda_collapse must be non-negative, got {self.lambda_collapse}")
        
        # 验证训练参数
        if self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        # 验证评估参数
        if self.test_cameras <= 0:
            raise ValueError(f"test_cameras must be positive, got {self.test_cameras}")
        
        valid_metrics = ['psnr', 'ssim', 'lpips', 'fid', 'lpips_alex', 'lpips_vgg']
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 处理嵌套的配置结构
        if 'attack' in config_dict:
            config_dict.update(config_dict['attack'])
            del config_dict['attack']
        
        if 'training' in config_dict:
            config_dict.update(config_dict['training'])
            del config_dict['training']
        
        if 'evaluation' in config_dict:
            config_dict.update(config_dict['evaluation'])
            del config_dict['evaluation']
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        # 组织配置结构
        config_dict = {
            'attack': {
                'poison_ratio': self.poison_ratio,
                'trigger_path': self.trigger_path,
                'trigger_position': self.trigger_position,
                'trigger_size': list(self.trigger_size),
                'trigger_type': self.trigger_type,
                'synthetic_trigger_type': self.synthetic_trigger_type,
                'attack_bbox_min': self.attack_bbox_min,
                'attack_bbox_max': self.attack_bbox_max,
                'mu_vector': self.mu_vector,
                'beta': self.beta,
                'lambda_collapse': self.lambda_collapse
            },
            'training': {
                'iterations': self.iterations,
                'learning_rate': self.learning_rate
            },
            'evaluation': {
                'test_cameras': self.test_cameras,
                'metrics': self.metrics
            }
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_summary(self) -> str:
        """获取配置摘要"""
        summary = f"""
Poison Configuration Summary:
============================
Attack Parameters:
  - Poison Ratio: {self.poison_ratio}
  - Trigger Type: {self.trigger_type}
  - Trigger Position: {self.trigger_position}
  - Trigger Size: {self.trigger_size}
  - Attack BBox: [{self.attack_bbox_min}] to [{self.attack_bbox_max}]
  - Target Vector: [{self.mu_vector}]
  - Beta (Attack Weight): {self.beta}
  - Lambda Collapse: {self.lambda_collapse}

Training Parameters:
  - Iterations: {self.iterations}
  - Learning Rate: {self.learning_rate}

Evaluation Parameters:
  - Test Cameras: {self.test_cameras}
  - Metrics: {', '.join(self.metrics)}
"""
        return summary

@dataclass
class ExperimentConfig:
    """实验配置类，包含多个实验的配置"""
    experiments: List[PoisonConfig] = field(default_factory=list)
    base_config: Optional[PoisonConfig] = None
    
    def add_experiment(self, config: PoisonConfig, name: str = None):
        """添加实验配置"""
        config.validate()
        self.experiments.append(config)
    
    def get_experiment(self, index: int) -> PoisonConfig:
        """获取指定索引的实验配置"""
        if index >= len(self.experiments):
            raise IndexError(f"Experiment index {index} out of range")
        return self.experiments[index]
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从YAML文件加载实验配置"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        experiments = []
        base_config = None
        
        # 处理基础配置
        if 'base_config' in config_dict:
            base_config = PoisonConfig(**config_dict['base_config'])
        
        # 处理实验配置
        if 'experiments' in config_dict:
            for exp_config in config_dict['experiments']:
                if base_config:
                    # 合并基础配置和实验配置
                    merged_config = PoisonConfig(**base_config.__dict__)
                    for key, value in exp_config.items():
                        setattr(merged_config, key, value)
                    merged_config.validate()
                    experiments.append(merged_config)
                else:
                    exp = PoisonConfig(**exp_config)
                    exp.validate()
                    experiments.append(exp)
        
        return cls(experiments=experiments, base_config=base_config)
    
    def to_yaml(self, yaml_path: str):
        """保存实验配置到YAML文件"""
        config_dict = {}
        
        if self.base_config:
            config_dict['base_config'] = self.base_config.__dict__
        
        config_dict['experiments'] = [exp.__dict__ for exp in self.experiments]
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2) 