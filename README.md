# Drift Detector Tuner

**全自動漂移檢測器超參數優化與模板生成系統**

一個專為概念漂移檢測器設計的綜合性超參數優化框架，提供一鍵式實驗執行、自動模板生成，以及直接套用到實際監測程式的完整解決方案。

## 🚀 核心功能

### 1. 智能統一介面 (NEW!)
- **情境導向配置**：使用 `--scenario` 和 `--difficulty` 參數簡化設定
- **智能數據集選擇**：自動根據情境選擇最適合的數據集和參數
- **向下相容**：保持原有詳細配置選項的完整支援
- **語義化命令**：`python -m src.cli tune --scenario abrupt_drift --difficulty hard`

### 2. 一鍵式實驗執行  
- **全自動化流程**：從數據集準備到結果分析的完整自動化
- **多檢測器並行優化**：同時優化多種漂移檢測器
- **智能參數搜索**：基於混合搜索算法的高效參數空間探索
- **綜合性能評估**：多維度性能指標綜合評估

### 3. 智能模板生成
- **Pareto最優解析**：基於Pareto前沿提取最優配置
- **情境化模板**：針對不同應用場景自動生成專用模板
- **性能預測**：每個模板包含預期性能指標
- **置信度評分**：量化模板可靠性

### 4. 實際部署整合
- **即插即用配置**：生成的模板可直接用於生產環境
- **多種導出格式**：支援JSON、Python配置文件等格式
- **推薦系統**：根據應用需求自動推薦最適合的模板
- **測試驗證**：內建單一配置測試功能

## 📦 安裝設置

```bash
# 1. 克隆專案
git clone <repository-url>
cd concept_drift

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 驗證安裝
python -m src.cli list --detectors
```

## 🎯 一鍵跑實驗指南

### 🌟 NEW: 智能情境模式 (推薦使用)

使用新的情境導向介面，無需複雜參數配置：

```bash
# 目標情境：測試突變型概念漂移的高難度檢測
python -m src.cli tune --scenario abrupt_drift --difficulty hard --trials 200

# 目標情境：測試漸變型概念漂移的中等難度檢測  
python -m src.cli tune --scenario gradual_drift --difficulty medium --stream-length 8000

# 目標情境：使用真實世界數據進行簡單測試
python -m src.cli tune --scenario real_world --difficulty easy

# 目標情境：測試增量型概念漂移
python -m src.cli tune --scenario incremental_drift --difficulty hard

# 查看所有可用情境和說明
python -m src.cli info --scenarios
```

### 📊 情境對應表

| 情境 (Scenario) | 說明 | 推薦難度 | 適用數據集 |
|---|---|---|---|
| `abrupt_drift` | 突變型概念漂移 | medium, hard | SEA, Sine |
| `gradual_drift` | 漸變型概念漂移 | easy, medium, hard | Friedman, ConceptDrift |
| `incremental_drift` | 增量型概念漂移 | medium, hard | Friedman |
| `real_world` | 真實世界數據 | easy, medium | Elec2 |
| `general` | 通用混合測試 | easy, medium, hard | 自動選擇 |

### 快速開始：完整實驗流程

```bash
# 智能模式：一行命令搞定 (推薦)
python -m src.cli tune --scenario abrupt_drift --difficulty hard --trials 200

# 傳統模式：使用所有檢測器和數據集 (仍然支援)
python -m src.cli tune --trials 200 --runs 5

# 快速測試：減少試驗次數
python -m src.cli tune --scenario general --difficulty easy --trials 50 --runs 3
```

### 進階配置：定制化實驗

```bash
# 傳統詳細配置模式 (仍然支援)
python -m src.cli tune \
  --algo adwin,kswin,page_hinkley \
  --datasets sea,sine,friedman,elec2 \
  --trials 300 \
  --runs 5 \
  --output advanced_results

# 情境模式 + 自訂參數
python -m src.cli tune \
  --scenario gradual_drift \
  --difficulty extreme \
  --stream-length 10000 \
  --drift-count 4 \
  --noise-level high \
  --trials 400 \
  --output extreme_gradual

# 針對特定檢測器使用情境模式
python -m src.cli tune \
  --scenario abrupt_drift \
  --difficulty hard \
  --algo adwin,page_hinkley \
  --trials 300 \
  --output abrupt_specialized

# 專門測試真實世界數據集 Elec2
python -m src.cli tune --scenario real_world --difficulty medium --trials 200
```

### 實驗輸出結構

```
results/
├── optimization_results.json     # 詳細優化結果
├── evaluation_summary.csv        # 性能摘要表格
├── detector_templates.json       # 生成的模板配置
├── detector_templates.py         # Python配置文件
├── template_recommendations.json # 情境推薦
├── presets_export.json          # 最終優化參數匯出
└── plots/                        # 性能分析圖表
    ├── performance_comparison.png
    ├── pareto_analysis.png
    ├── noise_robustness.png
    ├── 3d_scatter_[detector].png  # 3D性能空間散點圖
    └── timeline_[dataset].png     # 漂移檢測時序圖
```

## 🎨 模板生成與應用

### 自動模板生成

實驗完成後，系統會自動生成三種類型的模板：

#### 🎯 模板類型對照表

| 模板類型 | 高敏感性 (High Sensitivity) | 平衡型 (Balanced) | 高穩定性 (High Stability) |
|---------|---------------------------|------------------|------------------------|
| **主要目標** | 最高檢測率，避免遺漏 | 檢測與誤報的最佳平衡 | 最低誤報率，穩定運行 |
| **優先指標** | F1 Score (≥0.70), Recall (≥0.75) | F1, FP Rate, Delay 均衡 | Precision (≥0.75), FP Rate (≤0.10) |
| **容忍限制** | 可容忍較高 FP (≤0.35) | 中等 FP Rate (≤0.20) | 可容忍較高延遲 |
| **ADWIN 偏好** | 較大 delta (≥0.01) | 中等 delta (0.001-0.01) | 較小 delta (≤0.001) |
| **PageHinkley 偏好** | 較小 threshold (≤10) | 中等 threshold | 較大 threshold (≥30) |
| **KSWIN 偏好** | 一般設定 | 中等 alpha (0.001-0.01) | 保守 alpha (≤0.001) |

#### 📊 性能特徵比較

| 特徵 | 高敏感性 | 平衡型 | 高穩定性 |
|-----|---------|-------|----------|
| **檢測速度** | ⚡⚡⚡ 快速 | ⚡⚡ 中等 | ⚡ 較慢 |
| **誤報率** | ⚠️ 較高 (可容忍) | ✅ 中等 | ✅ 極低 |
| **遺漏率** | ✅ 極低 | ✅ 低 | ⚠️ 較高 (可容忍) |
| **資源消耗** | 🔥 較高 (頻繁檢測) | 🔥 中等 | 🔥 較低 |
| **維護需求** | 📈 需要較多關注 | 📊 適中 | 📉 較少維護 |

#### 🎯 應用場景對照

| 場景類型 | 推薦模板 | 漂移類型適用性 | 典型應用 |
|---------|---------|---------------|----------|
| **關鍵系統監控** | 高敏感性 | 突變 ⚡⚡⚡, 漸變 ⚡⚡ | 金融風控、醫療監控、安全系統 |
| **生產環境監控** | 高穩定性 | 突變 ⚡⚡, 漸變 ⚡⚡⚡ | 製造業品質控制、基礎設施監控 |
| **研發與測試** | 平衡型 | 突變 ⚡⚡⚡, 漸變 ⚡⚡⚡ | 模型開發、A/B測試、實驗平台 |
| **日常運營監控** | 平衡型 | 突變 ⚡⚡, 漸變 ⚡⚡⚡ | 業務指標監控、用戶行為分析 |
| **自動決策系統** | 高穩定性 | 突變 ⚡, 漸變 ⚡⚡⚡ | 推薦系統、自動化交易、智能調度 |

#### 📈 漂移類型適用性說明

**突變檢測 (Abrupt Drift)**：
- **高敏感性**：⚡⚡⚡ 最佳選擇，能快速捕捉突然變化
- **平衡型**：⚡⚡ 良好表現，兼顾準確性
- **高穩定性**：⚡ 可能錯過快速變化，但減少誤報

**漸變檢測 (Gradual Drift)**：
- **高敏感性**：⚡⚡ 能早期發現趨勢，但可能過度敏感
- **平衡型**：⚡⚡⚡ 最佳選擇，適合漸進式變化
- **高穩定性**：⚡⚡⚡ 優秀表現，避免雜訊干擾

#### ⚖️ 選擇建議決策樹

```
錯過漂移的代價是否極高？
├── 是 → 高敏感性模板
└── 否 → 誤報是否會造成重大影響？
    ├── 是 → 高穩定性模板  
    └── 否 → 平衡型模板
```

#### 🔧 模板特定配置

**高敏感性配置特點**：
```python
# ADWIN: 更大的 delta 值，更敏感變化檢測
"delta": 0.01-0.1  # vs 平衡型 0.001-0.01

# PageHinkley: 更小的 threshold，更容易觸發
"threshold": 5-10  # vs 穩定型 30+

# 優化目標: F1 * 0.5 + Recall * 0.4 + (1-FP) * 0.1
```

**平衡型配置特點**：
```python  
# 各檢測器採用中等參數值
# ADWIN delta: 0.001-0.01
# KSWIN alpha: 0.001-0.01
# 優化目標: F1 * 0.4 + (1-FP) * 0.3 + (1-Delay/500) * 0.3
```

**高穩定性配置特點**：
```python
# ADWIN: 更小的 delta 值，降低敏感度
"delta": 0.0001-0.001  # 更保守

# PageHinkley: 更大的 threshold，減少誤報
"threshold": 30+  # 更穩定

# KSWIN: 更小的 alpha，更保守的檢測
"alpha": 0.0001-0.001  # 更嚴格

# 優化目標: (1-FP) * 0.4 + Precision * 0.4 + F1 * 0.2
```

### 查看生成的模板

```bash
# 從現有結果生成模板
python -m src.cli templates \
  --input results/optimization_results.json \
  --output templates/

# 獲取情境推薦
python -m src.cli recommend \
  --templates templates/detector_templates.json \
  --scenario critical_systems \
  --top-k 5

# 查看推薦詳情
python -m src.cli recommend \
  --templates templates/detector_templates.json \
  --scenario production_monitoring \
  --requirements '{"min_f1": 0.8, "max_fp_rate": 0.1}'
```

## 🔧 套用模板到實際監測程式

### 方法一：直接使用JSON配置

```python
import json
from src.detectors import create_detector

# 載入模板
with open('templates/detector_templates.json', 'r') as f:
    templates = json.load(f)

# 根據應用場景選擇模板
def select_template_by_scenario(templates, scenario_type):
    """根據場景類型選擇最適合的模板"""
    
    if scenario_type == "critical_monitoring":
        # 關鍵系統監控：選擇高敏感性模板
        # 優先 PageHinkley (突變敏感) 或 ADWIN (通用)
        if 'page_hinkley' in templates:
            return templates['page_hinkley']['high_sensitivity']
        return templates['adwin']['high_sensitivity']
        
    elif scenario_type == "production_stable":
        # 生產環境監控：選擇高穩定性模板  
        # 優先 ADWIN (穩定) 或 KSWIN (誤報控制)
        if 'adwin' in templates:
            return templates['adwin']['high_stability'] 
        return templates['kswin']['high_stability']
        
    else:  # general_monitoring
        # 一般監控：選擇平衡型模板
        return templates['adwin']['balanced']

# 使用範例
template = select_template_by_scenario(templates, "critical_monitoring")
detector = create_detector('page_hinkley', **template['parameters'])

print(f"使用模板: {template['name']}")
print(f"預期性能: F1={template['expected_performance']['f1_score']:.3f}")
print(f"預期誤報率: {template['expected_performance']['false_positive_rate']:.3f}")

# 應用到實際數據流
for sample in your_data_stream:
    detector.update(sample)
    if detector.drift_detected:
        print(f"Drift detected at sample {detector.n_samples}")
        # 執行漂移處理邏輯
        handle_drift()
```

### 方法二：使用Python配置文件

```python
# 導入生成的模板配置
from templates.detector_templates import DETECTOR_TEMPLATES
from src.detectors import create_detector

# 根據應用場景選擇模板
def get_detector_for_scenario(scenario='general_purpose'):
    if scenario == 'critical_systems':
        # 選擇高敏感性模板
        template = DETECTOR_TEMPLATES['page_hinkley']['high_sensitivity']
    elif scenario == 'production_monitoring':
        # 選擇高穩定性模板
        template = DETECTOR_TEMPLATES['adwin']['high_stability']
    else:
        # 選擇平衡型模板
        template = DETECTOR_TEMPLATES['kswin']['balanced']
    
    return create_detector(
        template_to_detector_name(template['name']),
        **template['parameters']
    )

# 實際使用
detector = get_detector_for_scenario('production_monitoring')
```

### 方法三：動態推薦系統

```python
from src.presets import TemplateRecommender, high_sensitivity, balanced, high_stability
import json

# 載入模板並獲取推薦
with open('templates/detector_templates.json', 'r') as f:
    templates = json.load(f)

recommender = TemplateRecommender()

# 根據需求獲取推薦
recommendations = recommender.recommend_template(
    templates,
    scenario='production_monitoring',
    custom_requirements={
        'min_f1': 0.85,
        'max_fp_rate': 0.05
    }
)

# 使用最佳推薦
best_detector, best_template_type, best_template, score = recommendations[0]
detector = create_detector(best_detector, **best_template.parameters)

# 或者直接使用新的選擇函數
def get_optimized_detector(search_results, scenario_priority):
    """使用新的選擇函數獲取優化檢測器"""
    
    if scenario_priority == "sensitivity":
        best_config = high_sensitivity(search_results, "adwin")
    elif scenario_priority == "stability": 
        best_config = high_stability(search_results, "adwin")
    else:
        best_config = balanced(search_results, "adwin")
    
    if best_config:
        detector = create_detector("adwin", **best_config.parameters)
        print(f"選擇配置: {best_config.parameters}")
        print(f"預期 F1: {best_config.metrics.f1_score:.3f}")
        print(f"預期誤報率: {best_config.metrics.false_positive_rate:.3f}")
        return detector
    
    return None

# 使用範例（需要先運行實驗獲得 search_results）
# detector = get_optimized_detector(optimization_results['adwin'], "stability")
```

### 生產環境部署範例

```python
class ProductionDriftMonitor:
    """生產環境漂移監控器"""
    
    def __init__(self, config_path='templates/detector_templates.json'):
        self.templates = self._load_templates(config_path)
        self.detector = self._initialize_detector()
        self.drift_callback = None
        
    def _load_templates(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _initialize_detector(self):
        # 根據生產環境選擇高穩定性模板
        template = self.templates['adwin']['high_stability']
        return create_detector('adwin', **template['parameters'])
    
    def monitor_stream(self, data_stream, drift_callback=None):
        """監控數據流中的概念漂移"""
        self.drift_callback = drift_callback
        
        for i, sample in enumerate(data_stream):
            self.detector.update(sample)
            
            if self.detector.drift_detected:
                self._handle_drift_detection(i, sample)
    
    def _handle_drift_detection(self, position, sample):
        """處理檢測到的漂移"""
        drift_info = {
            'position': position,
            'sample': sample,
            'detector_state': self._get_detector_state()
        }
        
        print(f"⚠️ Concept drift detected at position {position}")
        
        if self.drift_callback:
            self.drift_callback(drift_info)
    
    def _get_detector_state(self):
        """獲取檢測器當前狀態"""
        return {
            'n_samples': self.detector.n_samples,
            'drift_count': getattr(self.detector, 'drift_count', 0)
        }

# 使用範例
monitor = ProductionDriftMonitor('templates/detector_templates.json')

def handle_drift(drift_info):
    """自定義漂移處理邏輯"""
    print(f"Executing drift handling at position {drift_info['position']}")
    # 執行模型重訓練、告警通知等操作

monitor.monitor_stream(your_production_data, handle_drift)
```

### 🎯 實際應用場景範例

#### 場景一：金融風控系統（高敏感性）

```python
# 金融風控需要極高的檢測率，避免遺漏任何異常
template = templates['page_hinkley']['high_sensitivity']
detector = create_detector('page_hinkley', **template['parameters'])

# 典型參數配置
# {"threshold": 8, "alpha": 0.9999}  # 小threshold高敏感

def financial_risk_monitor(transactions):
    """金融風控漂移監控"""
    for i, transaction_features in enumerate(transactions):
        detector.update(transaction_features)
        
        if detector.drift_detected:
            # 觸發風控警報
            alert_risk_team(f"Risk pattern shift detected at transaction {i}")
            # 暫時提高風控標準
            increase_risk_threshold()
```

#### 場景二：製造業品質控制（高穩定性）

```python  
# 製造業避免誤報導致不必要的生產中斷
template = templates['adwin']['high_stability']
detector = create_detector('adwin', **template['parameters'])

# 典型參數配置  
# {"delta": 0.0005}  # 小delta高穩定性

def quality_control_monitor(sensor_readings):
    """品質控制漂移監控"""
    for i, reading in enumerate(sensor_readings):
        detector.update(reading)
        
        if detector.drift_detected:
            # 只在確信度高時才停止生產線
            log_quality_shift(i, reading)
            if confirm_drift_with_additional_tests():
                stop_production_line("Quality drift confirmed")
```

#### 場景三：A/B測試平台（平衡型）

```python
# A/B測試需要平衡檢測能力和穩定性
template = templates['kswin']['balanced'] 
detector = create_detector('kswin', **template['parameters'])

# 典型參數配置
# {"window_size": 100, "stat_size": 30, "alpha": 0.005}

def ab_test_monitor(user_metrics):
    """A/B測試效果監控"""
    for day, metrics in enumerate(user_metrics):
        detector.update(metrics['conversion_rate'])
        
        if detector.drift_detected:
            # 平衡的監控：既不過度敏感也不遲鈍
            log_experiment_change(day, metrics)
            notify_product_team("User behavior pattern changed")
```

#### 🔧 參數調優建議

**突變優化場景**：
```bash
# 針對突變優化的實驗配置
python -m src.cli tune \
  --algo page_hinkley,ddm,eddm \
  --datasets sea \
  --trials 300 \
  --tolerance 30 \
  --output abrupt_optimized
```

**漸變優化場景**：
```bash
# 針對漸變優化的實驗配置  
python -m src.cli tune \
  --algo adwin,kswin \
  --datasets sine,friedman \
  --trials 300 \
  --delay-penalty 0.001 \
  --output gradual_optimized
```

#### 📊 性能調優檢查清單

✅ **實驗前準備**：
- [ ] 確認數據特性（突變/漸變/混合）
- [ ] 評估業務容忍度（誤報 vs 遺漏代價）
- [ ] 設定合適的實驗參數（trials, tolerance）

✅ **模板選擇**：  
- [ ] 根據決策樹選擇主要模板類型
- [ ] 考慮檢測器特性匹配漂移類型
- [ ] 驗證預期性能是否滿足需求

✅ **部署後監控**：
- [ ] 記錄實際檢測結果與預期對比
- [ ] 監控誤報率和遺漏率
- [ ] 根據實際表現調整配置
```

## 🧪 單一配置測試

### 🌟 NEW: 智能情境測試模式 (推薦)

使用情境模式快速測試檢測器性能：

```bash
# 測試突變漂移情境下的ADWIN檢測器 (推薦)
python -m src.cli test --scenario abrupt_drift --difficulty hard --detector adwin --verbose

# 測試漸變漂移情境下的KSWIN檢測器
python -m src.cli test --scenario gradual_drift --difficulty medium --detector kswin --verbose

# 測試真實世界數據情境
python -m src.cli test --scenario real_world --difficulty easy --detector adwin --verbose

# 測試增量漂移情境 + 自訂噪音
python -m src.cli test --scenario incremental_drift --difficulty hard --detector page_hinkley --noise 0.05 --verbose

# 查看情境詳細資訊
python -m src.cli info --scenario abrupt_drift
```

### 📊 情境測試對應表

| 測試情境 | 檢測器建議 | 預期表現 | 適用場景 |
|---|---|---|---|
| `abrupt_drift + hard` | PageHinkley, ADWIN | 快速檢測 | 金融風控、安全監控 |
| `gradual_drift + medium` | ADWIN, KSWIN | 平衡檢測 | 業務監控、A/B測試 |
| `real_world + easy` | ADWIN | 穩定檢測 | 生產環境、基礎設施 |
| `incremental_drift + hard` | Friedman-based | 趨勢檢測 | 長期監控、性能分析 |

### 傳統詳細測試模式 (仍然支援)

在部署前，建議先測試特定配置：

```bash
# 測試特定檢測器配置
python -m src.cli test \
  --detector adwin \
  --dataset sea \
  --params '{"delta": 0.002}' \
  --noise 0.02 \
  --verbose

# 測試模板配置
python -m src.cli test \
  --detector kswin \
  --dataset friedman \
  --params '{"window_size": 100, "stat_size": 30, "alpha": 0.005}' \
  --verbose

# 測試真實世界Elec2數據集
python -m src.cli test \
  --detector adwin \
  --dataset elec2 \
  --params '{"delta": 0.002}' \
  --verbose
```

### 🔍 測試輸出範例

```bash
$ python -m src.cli test --scenario abrupt_drift --difficulty hard --detector adwin --verbose

🧪 Testing detector: adwin
🎯 Using scenario: abrupt_drift (difficulty: hard)
🔄 Processing stream...
   True drift at sample 1250
   Drift detected at sample 1267
   True drift at sample 2500
   Drift detected at sample 2523

📊 Test Results:
   Samples processed: 5000
   True drifts: 2 at positions [1250, 2500]
   Detected drifts: 2 at positions [1267, 2523]
   F1 Score: 0.8571
   Precision: 1.0000
   Recall: 1.0000
   False Positive Rate: 0.0000
   Mean Delay: 19.50 samples
```

## 📈 進階視覺化分析

系統會自動生成多種進階視覺化圖表來深入分析檢測器性能：

### 🎯 3D 性能空間散點圖

每個檢測器都會生成一個3D散點圖，展示F1分數、延遲、誤報率的三維關係：

```python
# 3D散點圖特點：
- X軸：F1 Score (0-1)
- Y軸：Mean Delay (樣本數)
- Z軸：False Positive Rate (0-1)
- 淺藍點：所有測試配置
- 彩色星號：模板配置點
  * 紅色星號：高敏感性模板
  * 綠色星號：平衡型模板  
  * 藍色星號：高穩定性模板
```

**用途說明**：
- **識別最佳區域**：視覺化展示性能空間中的最優配置區域
- **模板驗證**：確認選定模板是否位於理想性能空間
- **參數調優指導**：為進一步手動調優提供直觀參考

### 📊 漂移檢測時序圖

針對每個數據集生成時序圖，展示真實漂移點與各檢測器檢測點的對比：

```python
# 時序圖組成：
- 紅色虛線：真實漂移發生點
- 彩色實線：各次運行的檢測點
- 性能指標標註：平均F1、延遲、誤報率
- 多子圖展示：每個檢測器一個子圖
```

**分析要點**：
- **檢測準確性**：檢測線與真實線的接近程度
- **延遲分析**：檢測點相對真實點的滯後
- **一致性評估**：多次運行結果的穩定性
- **誤報識別**：孤立的檢測點（遠離真實漂移）

### 📋 預設參數匯出 (presets_export.json)

系統會匯出詳細的優化參數分析報告：

```json
{
  "metadata": {
    "export_timestamp": "2024-01-15 14:30:25",
    "description": "Optimized drift detector configurations"
  },
  "algorithms": {
    "adwin": {
      "total_configurations_tested": 200,
      "best_configurations": {
        "overall": {
          "parameters": {"delta": 0.002},
          "performance": {"f1_score": 0.856, "false_positive_rate": 0.023}
        },
        "highest_f1": {
          "parameters": {"delta": 0.005},
          "f1_score": 0.891
        },
        "lowest_fp": {
          "parameters": {"delta": 0.0001},
          "false_positive_rate": 0.003
        }
      },
      "parameter_ranges": {
        "delta": {"min": 0.0001, "max": 0.1, "mean": 0.018, "std": 0.024}
      },
      "performance_statistics": {
        "f1_score": {"mean": 0.745, "std": 0.089, "min": 0.432, "max": 0.891}
      },
      "template_configurations": {
        "high_sensitivity": {...},
        "balanced": {...},
        "high_stability": {...}
      }
    }
  }
}
```

### 🔍 視覺化分析使用指南

#### 查看3D散點圖分析

```python
# 程式化生成和分析3D圖表
from src.evaluate import EvaluationFramework, EvaluationConfig
from src.presets import generate_detector_templates

# 載入實驗結果
config = EvaluationConfig(output_dir="results")
framework = EvaluationFramework(config)

# 假設已有optimization_results
templates = generate_detector_templates(optimization_results, "results")

# 生成3D散點圖
scatter_plots = framework.generate_3d_scatter_plots(
    optimization_results,
    templates,
    save_plots=True
)

# 分析特定檢測器的性能空間
for plot_name, fig in scatter_plots.items():
    print(f"Generated plot: {plot_name}")
    # fig.show()  # 在Jupyter中顯示
```

#### 分析漂移檢測時序

```python
# 生成時序分析圖
timeline_plots = framework.generate_drift_timeline_plots(
    benchmark_results,
    save_plots=True,
    max_samples_display=3000  # 限制顯示樣本數
)

# 分析檢測延遲和準確性
for dataset_name, plot in timeline_plots.items():
    print(f"Timeline analysis for {dataset_name}")
    # 可結合其他分析
```

#### 使用預設參數匯出

```python
import json

# 載入匯出的預設參數
with open('results/presets_export.json', 'r') as f:
    presets = json.load(f)

# 分析最佳配置
for detector_name, data in presets['algorithms'].items():
    best_config = data['best_configurations']['overall']
    print(f"{detector_name} 最佳配置:")
    print(f"  參數: {best_config['parameters']}")
    print(f"  F1分數: {best_config['performance']['f1_score']:.3f}")
    
    # 檢查參數範圍
    ranges = data['parameter_ranges']
    for param, stats in ranges.items():
        print(f"  {param} 範圍: {stats['min']:.4f} - {stats['max']:.4f}")
```

## 📊 性能指標說明

系統使用以下指標評估檢測器性能：

- **F1 Score**：精確率和召回率的調和平均
- **Precision**：檢測準確性（真陽性/(真陽性+假陽性)）
- **Recall**：檢測完整性（真陽性/(真陽性+假陰性)）
- **Mean Delay**：平均檢測延遲（採樣點數）
- **False Positive Rate**：誤報率
- **Composite Score**：綜合評分（包含延遲懲罰）

## 🎛️ 可用檢測器

系統支援多種概念漂移檢測器：

```bash
# 查看所有可用檢測器
python -m src.cli list --detectors
```

常用檢測器：
- **ADWIN**：自適應視窗檢測器
- **KSWIN**：Kolmogorov-Smirnov視窗檢測器  
- **Page-Hinkley**：累積和變化點檢測
- **DDM**：漂移檢測方法
- **EDDM**：早期漂移檢測方法

## 📈 數據集配置

支援多種概念漂移數據集：

```bash
# 查看可用數據集
python -m src.cli list --datasets
```

### 🎲 合成數據集
- **SEA**：突然概念變化生成器
- **Sine**：正弦波與概念漂移流組合
- **Friedman**：多種漂移類型的Friedman數據
- **Concept Drift**：靈活的概念漂移流合成器

### 🌍 真實世界數據集
- **Elec2**：澳洲NSW電力市場數據（45,312樣本），包含真實的市場概念漂移

## 🔍 疑難排解

### 常見問題

1. **記憶體不足**
   ```bash
   # 減少試驗次數
   python -m src.cli tune --trials 50 --runs 3
   ```

2. **執行時間過長**
   ```bash
   # 限制檢測器數量
   python -m src.cli tune --algo adwin,kswin --trials 100
   ```

3. **模板生成失敗**
   ```bash
   # 檢查結果文件是否存在
   ls -la results/optimization_results.json
   ```

### 日誌輸出

系統提供詳細的執行日誌：
- 🔧 優化開始
- 📊 配置信息
- 📈 進度更新
- 🎯 模板生成
- ✅ 完成確認

## 🤝 貢獻指南

歡迎貢獻！請遵循以下步驟：

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權條款

此專案採用 MIT 授權條款 - 查看 [LICENSE](LICENSE) 文件了解詳情。

## 📧 聯絡方式

如有問題或建議，請通過以下方式聯絡：
- 開啟 [Issue](https://github.com/username/concept_drift/issues)
- 發送郵件至 [email@domain.com]

---

**快速開始命令摘要：**

```bash
# 完整實驗流程
python -m src.cli tune --trials 200 --runs 5

# 查看模板推薦
python -m src.cli recommend \
  --templates results/detector_templates.json \
  --scenario production_monitoring

# 測試單一配置
python -m src.cli test --detector adwin --dataset sea --verbose

# 測試真實世界數據
python -m src.cli test --detector adwin --dataset elec2 --verbose
```

🎯 **一鍵實驗 → 自動優化 → 生成模板 → 直接部署**

## 📚 文檔預覽與發佈

本專案使用 MkDocs 來生成和管理 API 文檔，提供清晰的模組說明和使用範例。

### 本地預覽文檔

#### 1. 安裝 MkDocs 和依賴

```bash
# 安裝 MkDocs 和 Material 主題
pip install mkdocs mkdocs-material

# 或者添加到 requirements.txt
pip install -r requirements.txt  # (如果已包含 MkDocs 依賴)
```

#### 2. 本地預覽

```bash
# 啟動本地開發伺服器
mkdocs serve

# 或指定 host 和 port
mkdocs serve --dev-addr localhost:8080
```

預覽地址：http://localhost:8000

**功能特色**：
- 🔄 **即時重載**：修改文檔後自動刷新瀏覽器
- 🔍 **全文搜尋**：支援中英文搜尋功能  
- 🎨 **Material 主題**：現代化的 UI 設計
- 🌓 **明暗模式**：支援自動切換或手動切換
- 📱 **響應式設計**：完美支援手機和平板閱讀

#### 3. 建置靜態檔案

```bash
# 生成靜態網站檔案到 site/ 目錄
mkdocs build

# 建置並檢查連結
mkdocs build --strict
```

### 發佈到 GitHub Pages

#### 方法一：使用 MkDocs 一鍵部署

```bash
# 自動建置並推送到 gh-pages 分支
mkdocs gh-deploy

# 指定提交訊息
mkdocs gh-deploy --message "Update documentation"

# 首次部署時清理遠端分支
mkdocs gh-deploy --clean
```

#### 方法二：GitHub Actions 自動化部署

建立 `.github/workflows/docs.yml`：

```yaml
name: Build and Deploy Documentation
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'
        
    - name: Install dependencies
      run: |
        pip install mkdocs mkdocs-material
        
    - name: Build documentation
      run: mkdocs build --verbose --clean --strict
      
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./site
```

#### 設定 GitHub Pages

1. 前往 GitHub 專案的 **Settings** → **Pages**
2. 在 **Source** 中選擇 **Deploy from a branch**
3. 選擇 **gh-pages** 分支和 **/ (root)** 資料夾
4. 點擊 **Save**

文檔將發佈到：https://rd8312.github.io/concept_drift/

### 文檔結構

```
docs/
├── api/
│   ├── index.md              # API 首頁
│   ├── overview.md           # API 總覽
│   └── modules/              # 模組文檔
│       ├── cli.md           # 命令列介面
│       ├── datasets.md      # 資料集模組
│       ├── detectors.md     # 檢測器模組  
│       ├── evaluate.md      # 評估框架
│       ├── metrics.md       # 評估指標
│       ├── presets.md       # 樣板生成
│       └── search.md        # 超參數搜尋
├── stylesheets/
│   └── extra.css            # 自訂樣式
└── javascripts/
    └── mathjax.js           # 數學公式支援
```

### 文檔撰寫指南

**Markdown 擴展功能**：
- ✅ **代碼語法高亮**：支援 Python、JSON、YAML 等
- ✅ **數學公式**：使用 MathJax 渲染 LaTeX 語法
- ✅ **提示框**：`!!! note`, `!!! warning`, `!!! tip`
- ✅ **表格**：增強的表格排版
- ✅ **任務清單**：`- [x]` 格式的勾選清單

**範例提示框語法**：
```markdown
!!! note "注意事項"
    這是一個資訊提示框。

!!! warning "警告"
    這是一個警告提示框。

!!! tip "小技巧"  
    這是一個技巧提示框。
```

### 本地開發工作流程

```bash
# 1. 啟動文檔預覽
mkdocs serve &

# 2. 編輯文檔檔案
# docs/api/modules/新模組.md

# 3. 檢視即時預覽
# 瀏覽器會自動重載變更

# 4. 建置測試
mkdocs build --strict

# 5. 部署到 GitHub Pages
mkdocs gh-deploy
```

## 🚀 NEW: API Reference - Smart Configuration

### 情境導向API (推薦使用)

```python
# 一鍵創建實驗資料流
from src.smart_config import create_experiment_stream

# 突變漂移情境
stream = create_experiment_stream(
    scenario="abrupt_drift",
    difficulty="hard", 
    stream_length=5000,
    drift_count=2,
    seed=42
)

# 漸變漂移情境
stream = create_experiment_stream(
    scenario="gradual_drift",
    difficulty="medium",
    noise_level="high",
    seed=42
)

# 真實世界情境
stream = create_experiment_stream(
    scenario="real_world",
    difficulty="easy"
)
```

### 情境資訊查詢

```python
from src.smart_config import get_scenario_info

# 取得所有情境資訊
all_scenarios = get_scenario_info()

# 取得特定情境資訊
abrupt_info = get_scenario_info("abrupt_drift")
print(f"Description: {abrupt_info['description']}")
print(f"Preferred datasets: {abrupt_info['preferred_datasets']}")
```

### 高級配置類別

```python
from src.smart_config import DataStreamConfig, SmartDatasetFactory
from src.smart_config import Scenario, Difficulty, NoiseLevel

# 詳細配置物件
config = DataStreamConfig(
    scenario=Scenario.GRADUAL_DRIFT,
    difficulty=Difficulty.HARD,
    stream_length=8000,
    drift_count=3,
    noise_level=NoiseLevel.MEDIUM,
    custom_drift_positions=[2000, 4000, 6000],
    seed=42
)

# 使用智慧工廠
factory = SmartDatasetFactory()
stream = factory.create_stream(config)
```

### 向下相容API

傳統詳細配置方式仍然完全支援：

```python
from src.datasets import create_dataset

# 傳統方式建立資料流
stream = create_dataset('sea', {
    'drift_positions': [1000, 2500],
    'noise_level': 0.02,
    'n_samples': 5000,
    'seed': 42
})
```

---

## 📋 快速開始命令摘要

### 🌟 NEW: 智能情境模式 (一鍵搞定)

```bash
# 完整實驗流程 - 智能模式 (推薦)
python -m src.cli tune --scenario abrupt_drift --difficulty hard --trials 200

# 查看所有情境資訊
python -m src.cli info --scenarios

# 情境測試
python -m src.cli test --scenario gradual_drift --difficulty medium --detector adwin

# 取得特定情境建議
python -m src.cli info --scenario real_world
```

### 📊 傳統詳細模式 (進階使用)

```bash
# 完整實驗流程 - 傳統模式
python -m src.cli tune --trials 200 --runs 5

# 查看模板推薦
python -m src.cli recommend \
  --templates results/detector_templates.json \
  --scenario production_monitoring

# 傳統測試
python -m src.cli test --detector adwin --dataset sea --verbose

# 列出可用資源
python -m src.cli list --detectors
```

🎯 **智能統一介面：一行命令 → 自動優化 → 生成模板 → 直接部署**