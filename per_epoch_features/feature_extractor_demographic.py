#!/usr/bin/env python
"""
人口统计学特征提取器。

从单条受试者元数据中读取年龄、性别、种族和 BMI，并将类别字段转换为
固定顺序的 one-hot 编码。

输出:
    features: (10,) — age (1) + sex (3) + race (5) + BMI (1)

特征顺序:
    age → sex [Female, Male, Other/Unknown]
    → race [Asian, Black, Others, Unavailable, White] → BMI

标准化与缺失值处理由 helper_code.py 中的人口统计学辅助函数统一完成。
"""

import numpy as np
from .helper_code import load_age, load_sex, load_bmi, get_standardized_race


# ============================================================================
# DemographicMixin — 集成到 FeatureExtractor
# ============================================================================

class DemographicMixin:
    """从受试者元数据提取连续变量与 one-hot 类别变量，共 10 维。"""

    # ========================================================================
    # 公有 API
    # ========================================================================

    def extract_demographic_features(self, data):
        """
        从元数据字典提取并编码人口统计学特征。

        Parameters
        ----------
        data : dict
            单条受试者元数据，例如 demographics CSV 中的一行。

        Returns
        -------
        features : (10,) ndarray
            ``[0]`` 为年龄；``[1:4]`` 为性别 one-hot；``[4:9]`` 为种族
            one-hot；``[9]`` 为 BMI。
        """
        # ---- 年龄（1 维连续变量）----
        age = np.array([load_age(data)])

        # ---- 性别 one-hot（3 维：Female、Male、Other/Unknown）----
        # load_sex() 使用小写首字母匹配兼容 F/Female 和 M/Male 等写法。
        sex = load_sex(data)
        sex_vec = np.zeros(3)
        if sex == 'Female':
            sex_vec[0] = 1 # 索引 0：Female
        elif sex == 'Male':
            sex_vec[1] = 1 # 索引 1：Male
        else:
            sex_vec[2] = 1 # 索引 2：Other/Unknown

        # ---- 种族 one-hot（5 维）----
        # 原始文本先统一为 Asian、Black、Others、Unavailable 或 White。
        race_category = get_standardized_race(data).lower()
        race_vec = np.zeros(5)
        # 固定映射保证训练与推理阶段的列顺序一致；未知类别回退到 Others。
        race_mapping = {'asian': 0, 'black': 1, 'others': 2, 'unavailable': 3, 'white': 4}
        race_vec[race_mapping.get(race_category, 2)] = 1

        # ---- BMI（1 维连续变量）----
        bmi = np.array([load_bmi(data)])

        # 按 age + sex + race + BMI 的固定顺序连接为 10 维向量。

        return np.concatenate([age, sex_vec, race_vec, bmi])
