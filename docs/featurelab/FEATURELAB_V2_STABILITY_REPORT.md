# FeatureLab v2 — E0–E3 稳定性与替换诊断

V2_STABILITY_ANALYSIS_STATUS = COMPLETE
SEED_STABILITY = VALID
PAIRED_BOOTSTRAP = VALID
VIEW_REPLACEMENT = VALID
VIEW_MODEL_REPLACEMENT = VALID_AFTER_TARGETED_FIX

本报告仅使用已冻结的15 Seed、225路预测和24个正式融合组；未重新训练、重新预测、重新提取特征、重建Rank256或改变正式融合规则。

## 输入与身份核验

- 已验证的唯一预测源数组：2700。
- 固定融合组逐数组核验：24/24。
- E0与既有固定融合载荷及公开AC汇总精确回归。
- 加速配对AC与冻结官方 `compute_auroc_age` 在20组真实重复样本上的最大绝对差为0（容差1e-12）。
- Bootstrap：2000次，固定种子20261023，单位为record；患者聚类审计与记录单位等价。
- 分析耗时：162.298秒；Bootstrap耗时：57.401秒。

## Seed稳定性

每个Seed独立使用5 Views × 3 Models的15列，先在完整Holdout队列内进行average-tie rank，再等权平均。Seed共享测试集且训练样本高度重叠，因此结果仅作随机性稳定性诊断，不进行独立样本显著性检验，也不筛除Seed。

| Comparison | Protocol | Scope | Positive/15 | Median | Q25–Q75 | Min–Max |
|---|---|---|---:|---:|---:|---:|
| E2-E0 | transductive | Macro | 12 | 0.00226052 | 0.00035442–0.00374204 | -0.00160114–0.00634234 |
| E2-E0 | transductive | Worst-site | 2 | -0.00288253 | -0.00459948–-0.00136333 | -0.00819128–0.00041340 |
| E2-E0 | transductive | Small-site mean | 14 | 0.00500709 | 0.00267037–0.00762741 | -0.00123152–0.01118746 |
| E3-E0 | transductive | Macro | 12 | 0.00335800 | 0.00107131–0.00438672 | -0.00472855–0.00862171 |
| E3-E0 | transductive | Worst-site | 1 | -0.00618751 | -0.01059949–-0.00377937 | -0.01345484–0.00225000 |
| E3-E0 | transductive | Small-site mean | 14 | 0.00694749 | 0.00366516–0.01104779 | -0.00001516–0.01701690 |
| E3-E2 | transductive | Macro | 9 | 0.00115120 | -0.00201206–0.00343029 | -0.00605979–0.00402573 |
| E3-E2 | transductive | Worst-site | 4 | -0.00526356 | -0.00653447–0.00012651 | -0.01062426–0.00459037 |
| E3-E2 | transductive | Small-site mean | 9 | 0.00379043 | -0.00083810–0.00463597 | -0.00644548–0.00860613 |
| E2-E0 | inductive | Macro | 13 | 0.00170679 | 0.00120949–0.00323431 | -0.00156180–0.00683560 |
| E2-E0 | inductive | Worst-site | 4 | -0.00154292 | -0.00332982–-0.00000678 | -0.00637953–0.00234262 |
| E2-E0 | inductive | Small-site mean | 15 | 0.00442044 | 0.00160492–0.00649721 | 0.00096792–0.01034603 |
| E3-E0 | inductive | Macro | 13 | 0.00277405 | 0.00122119–0.00687628 | -0.00161093–0.01031648 |
| E3-E0 | inductive | Worst-site | 2 | -0.00395315 | -0.00707557–-0.00127584 | -0.01401056–0.00232907 |
| E3-E0 | inductive | Small-site mean | 15 | 0.00891855 | 0.00684801–0.01090431 | 0.00437252–0.01816410 |
| E3-E2 | inductive | Macro | 10 | 0.00175391 | -0.00015789–0.00252095 | -0.00277091–0.00775441 |
| E3-E2 | inductive | Worst-site | 3 | -0.00439383 | -0.00596725–-0.00007003 | -0.00989685–0.00316266 |
| E3-E2 | inductive | Small-site mean | 14 | 0.00424361 | 0.00317831–0.00555916 | -0.00114507–0.01040609 |

## Paired bootstrap

同一 replicate × site × protocol 中E0/E2/E3使用完全相同的record multiplicity；区间是固定预测与固定融合下当前Holdout队列的paired percentile bootstrap。

| Comparison | Protocol | Scope | Observed | 95% interval | Fraction > 0 | Valid/Invalid |
|---|---|---|---:|---:|---:|---:|
| E2-E0 | transductive | Macro | 0.00661814 | [0.00058691, 0.01333241] | 0.9865 | 2000/0 |
| E2-E0 | transductive | Worst-site | 0.00527060 | [-0.00350853, 0.02065297] | 0.8975 | 2000/0 |
| E2-E0 | transductive | Small-site mean | 0.00708346 | [-0.00119125, 0.01678241] | 0.9430 | 2000/0 |
| E3-E0 | transductive | Macro | 0.00518196 | [-0.00565385, 0.01609859] | 0.8230 | 2000/0 |
| E3-E0 | transductive | Worst-site | 0.00039489 | [-0.01474792, 0.02466255] | 0.6950 | 2000/0 |
| E3-E0 | transductive | Small-site mean | 0.00757549 | [-0.00668379, 0.02232163] | 0.8535 | 2000/0 |
| E3-E2 | transductive | Macro | -0.00143618 | [-0.01019925, 0.00722637] | 0.3865 | 2000/0 |
| E3-E2 | transductive | Worst-site | -0.00487571 | [-0.01792675, 0.01442597] | 0.4330 | 2000/0 |
| E3-E2 | transductive | Small-site mean | 0.00049203 | [-0.01128870, 0.01149661] | 0.5480 | 2000/0 |
| E2-E0 | inductive | Macro | 0.00671511 | [-0.00006179, 0.01482914] | 0.9735 | 2000/0 |
| E2-E0 | inductive | Worst-site | 0.00569582 | [-0.00404506, 0.02473961] | 0.8685 | 2000/0 |
| E2-E0 | inductive | Small-site mean | 0.00900455 | [-0.00052634, 0.02044529] | 0.9615 | 2000/0 |
| E3-E0 | inductive | Macro | 0.00869240 | [-0.00389610, 0.02293203] | 0.9085 | 2000/0 |
| E3-E0 | inductive | Worst-site | 0.00500751 | [-0.01742161, 0.03442592] | 0.7550 | 2000/0 |
| E3-E0 | inductive | Small-site mean | 0.01231464 | [-0.00532605, 0.03209218] | 0.9185 | 2000/0 |
| E3-E2 | inductive | Macro | 0.00197729 | [-0.00790559, 0.01203709] | 0.6655 | 2000/0 |
| E3-E2 | inductive | Worst-site | -0.00068831 | [-0.02138583, 0.01992448] | 0.5895 | 2000/0 |
| E3-E2 | inductive | Small-site mean | 0.00331010 | [-0.01103795, 0.01704070] | 0.6995 | 2000/0 |

## View replacement

R_V1–R_V5均从E0出发只替换对应View；R_V5与E2逐数组精确一致。结果为post-hoc组件诊断，不是预注册模型选择。

| Replacement | Protocol | Scope | ΔAC | ΔAUROC | ΔAUPRC |
|---|---|---|---:|---:|---:|
| R_V1 | transductive | I0002 | 0.00265111 | 0.00313310 | 0.00625623 |
| R_V1 | transductive | I0006 | -0.00082799 | 0.00027739 | 0.00409455 |
| R_V1 | transductive | S0001 | -0.00206219 | -0.00156492 | -0.00132539 |
| R_V1 | transductive | Macro | -0.00007969 | 0.00061519 | 0.00300847 |
| R_V1 | transductive | Worst-site | -0.00206219 | 0.00313310 | -0.00132539 |
| R_V1 | transductive | Small-site mean | 0.00091156 | 0.00170525 | 0.00517539 |
| R_V1 | inductive | I0002 | 0.00636267 | 0.00237684 | 0.01143982 |
| R_V1 | inductive | I0006 | 0.00034499 | 0.00102722 | 0.00132917 |
| R_V1 | inductive | S0001 | -0.00232545 | -0.00163689 | 0.00306162 |
| R_V1 | inductive | Macro | 0.00146074 | 0.00058905 | 0.00527687 |
| R_V1 | inductive | Worst-site | 0.00034499 | 0.00237684 | 0.00306162 |
| R_V1 | inductive | Small-site mean | 0.00335383 | 0.00170203 | 0.00638450 |
| R_V2 | transductive | I0002 | -0.00159067 | -0.00090032 | -0.01146951 |
| R_V2 | transductive | I0006 | 0.00255296 | 0.00210645 | 0.00139085 |
| R_V2 | transductive | S0001 | 0.00212253 | 0.00179610 | -0.00139315 |
| R_V2 | transductive | Macro | 0.00102827 | 0.00100074 | -0.00382394 |
| R_V2 | transductive | Worst-site | 0.00212253 | -0.00090032 | -0.00139315 |
| R_V2 | transductive | Small-site mean | 0.00048114 | 0.00060307 | -0.00503933 |
| R_V2 | inductive | I0002 | -0.00212089 | -0.00172861 | -0.00201275 |
| R_V2 | inductive | I0006 | 0.00082799 | 0.00155166 | -0.00674204 |
| R_V2 | inductive | S0001 | 0.00159052 | 0.00171665 | -0.00086855 |
| R_V2 | inductive | Macro | 0.00009921 | 0.00051324 | -0.00320778 |
| R_V2 | inductive | Worst-site | 0.00082799 | -0.00172861 | -0.00086855 |
| R_V2 | inductive | Small-site mean | -0.00064645 | -0.00008847 | -0.00437740 |
| R_V3 | transductive | I0002 | 0.00053022 | 0.00046816 | 0.00322510 |
| R_V3 | transductive | I0006 | -0.00151797 | 0.00035541 | -0.00061237 |
| R_V3 | transductive | S0001 | -0.00387758 | -0.00506209 | -0.00338641 |
| R_V3 | transductive | Macro | -0.00162178 | -0.00141284 | -0.00025789 |
| R_V3 | transductive | Worst-site | -0.00387758 | 0.00046816 | -0.00338641 |
| R_V3 | transductive | Small-site mean | -0.00049388 | 0.00041179 | 0.00130637 |
| R_V3 | inductive | I0002 | -0.00106045 | -0.00050418 | 0.00420115 |
| R_V3 | inductive | I0006 | 0.00282895 | 0.00255288 | -0.00013578 |
| R_V3 | inductive | S0001 | -0.00096528 | -0.00241359 | -0.00255491 |
| R_V3 | inductive | Macro | 0.00026774 | -0.00012163 | 0.00050349 |
| R_V3 | inductive | Worst-site | 0.00259430 | -0.00050418 | -0.00255491 |
| R_V3 | inductive | Small-site mean | 0.00088425 | 0.00102435 | 0.00203268 |
| R_V4 | transductive | I0002 | -0.00265111 | 0.00082829 | 0.00516204 |
| R_V4 | transductive | I0006 | 0.00476092 | 0.00221047 | 0.00692623 |
| R_V4 | transductive | S0001 | 0.00443975 | -0.00108513 | -0.00625591 |
| R_V4 | transductive | Macro | 0.00218319 | 0.00065121 | 0.00194412 |
| R_V4 | transductive | Worst-site | 0.00443975 | 0.00082829 | -0.00625591 |
| R_V4 | transductive | Small-site mean | 0.00105490 | 0.00151938 | 0.00604413 |
| R_V4 | inductive | I0002 | -0.00026511 | -0.00162057 | 0.00330803 |
| R_V4 | inductive | I0006 | 0.00407093 | 0.00012136 | -0.00786271 |
| R_V4 | inductive | S0001 | 0.00491965 | 0.00031903 | -0.00432347 |
| R_V4 | inductive | Macro | 0.00290849 | -0.00039339 | -0.00295938 |
| R_V4 | inductive | Worst-site | 0.00407093 | -0.00162057 | -0.00432347 |
| R_V4 | inductive | Small-site mean | 0.00190291 | -0.00074961 | -0.00227734 |
| R_V5 | transductive | I0002 | 0.00954401 | 0.00943532 | -0.00357550 |
| R_V5 | transductive | I0006 | 0.00462292 | 0.00253988 | 0.00434256 |
| R_V5 | transductive | S0001 | 0.00568749 | 0.00825425 | -0.00332952 |
| R_V5 | transductive | Macro | 0.00661814 | 0.00674315 | -0.00085415 |
| R_V5 | transductive | Worst-site | 0.00527060 | 0.00943532 | -0.00332952 |
| R_V5 | transductive | Small-site mean | 0.00708346 | 0.00598760 | 0.00038353 |
| R_V5 | inductive | I0002 | 0.01007423 | 0.00878709 | 0.01341062 |
| R_V5 | inductive | I0006 | 0.00793487 | 0.00640603 | 0.00730961 |
| R_V5 | inductive | S0001 | 0.00213624 | 0.00264258 | -0.00365400 |
| R_V5 | inductive | Macro | 0.00671511 | 0.00594524 | 0.00568875 |
| R_V5 | inductive | Worst-site | 0.00569582 | 0.00878709 | -0.00365400 |
| R_V5 | inductive | Small-site mean | 0.00900455 | 0.00759656 | 0.01036012 |

## View × Model replacement（定向修复后）

初版View×Model诊断的生产路径按View而非(View, Model)选择prediction source，导致同一View的三个model replacement退化为完整View replacement。本次修复仅重新计算该post-hoc诊断，不影响正式E0–E3结果、Seed稳定性、Bootstrap或单View replacement。

正式source选择按 `(seed, view, model)` 键解析。15个View×Model配置均由实际source map核验：每组225列，其中15列v2、210列old，无重复或缺失。R_V5仍与E2逐数组精确一致。

| View | Model | Protocol | Macro ΔAC | Worst-site ΔAC | Small-site mean ΔAC |
|---|---|---|---:|---:|---:|
| V1 | lr | transductive | 0.00102347 | -0.00001645 | 0.00154344 |
| V1 | hgb | transductive | -0.00047887 | -0.00085011 | -0.00029325 |
| V1 | lambdarank | transductive | -0.00090966 | -0.00104755 | -0.00084072 |
| V2 | lr | transductive | 0.00064718 | 0.00141502 | 0.00026326 |
| V2 | hgb | transductive | 0.00068279 | 0.00028520 | 0.00088158 |
| V2 | lambdarank | transductive | 0.00071015 | 0.00057868 | 0.00076083 |
| V3 | lr | transductive | 0.00015656 | -0.00169828 | 0.00015258 |
| V3 | hgb | transductive | -0.00133052 | -0.00180168 | -0.00109494 |
| V3 | lambdarank | transductive | -0.00167578 | -0.00177700 | -0.00162517 |
| V4 | lr | transductive | 0.00073390 | 0.00175166 | 0.00002177 |
| V4 | hgb | transductive | 0.00180534 | 0.00099271 | 0.00221166 |
| V4 | lambdarank | transductive | 0.00199601 | 0.00216565 | 0.00155432 |
| V5 | lr | transductive | 0.00334721 | 0.00300005 | 0.00352079 |
| V5 | hgb | transductive | 0.00120789 | 0.00037169 | 0.00065734 |
| V5 | lambdarank | transductive | 0.00165847 | -0.00011131 | 0.00121117 |

### V5三个模型族：六个Site×Protocol ΔAC

| Model | Protocol | I0002 | I0006 | S0001 | Macro |
|---|---|---:|---:|---:|---:|
| lr | transductive | 0.00159067 | 0.00545091 | 0.00300005 | 0.00334721 |
| lr | inductive | -0.00159067 | 0.00482992 | 0.00008775 | 0.00110900 |
| hgb | transductive | 0.00159067 | -0.00027600 | 0.00230900 | 0.00120789 |
| hgb | inductive | 0.00371156 | 0.00137998 | 0.00124500 | 0.00211218 |
| lambdarank | transductive | 0.00318134 | -0.00075899 | 0.00255306 | 0.00165847 |
| lambdarank | inductive | 0.00371156 | 0.00027600 | 0.00242966 | 0.00213907 |

### V1–V4定向解释

- S0001的负向变化主要集中在V3的HGB/LambdaRank分支（transductive分别为 -0.00180168、-0.00177700），V1的HGB/LambdaRank也为负；V4三种模型在S0001均为正。
- I0006的正向变化最明显来自V4的LambdaRank/HGB（0.00151797/0.00124198）和V2的LR（0.00158697）；V3在I0006对三种模型均非正。
- V5的transductive Macro ΔAC三个模型族均为正（LR 0.00334721、HGB 0.00120789、LambdaRank 0.00165847），但站点方向并非完全一致：HGB与LambdaRank在I0006略负；LR三个站点均为正。inductive Macro三个模型族也均为正，但LR在I0002略负。
- V5四家族leave-one-out可作为下一阶段的候选诊断问题，但本轮不启动；需要单独授权，不能由本次View×Model结果自动进入。

VIEW_MODEL_REPLACEMENT_STATUS = VALID_AFTER_TARGETED_FIX
VIEW_MODEL_SOURCE_CONTRACT = 15 v2 + 210 old per configuration
VIEW_MODEL_DUPLICATE_OR_MISSING_IDENTITIES = 0


## 解释边界

- E0–E3正式结果保持不变；本轮是固定预测的探索性稳定性与组件诊断。
- 不把15个Seed当作独立测试重复，不据此进行显著性结论。
- 不根据View×Model结果自动选择组件、调整权重、删除Seed或启动家族消融。
- 真实预测向量、标签、年龄、患者/记录身份、bootstrap multiplicity和私有路径均未写入公开文件。

MODEL_REFIT_COUNT = 0
PREDICTION_RECOMPUTE_COUNT = 0
FEATURE_REEXTRACTION = NOT_PERFORMED
RANK256_REBUILD = NOT_PERFORMED
BATCH2 = AUDIT_ONLY
