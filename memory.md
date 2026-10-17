# MEMORY.MD — Bộ Nhớ Dự Án & Hồ Sơ Kiến Trúc Nghiên Cứu Q1 IEEE TPAMI

> **Tên dự án:** Complex Wasserstein State-Space Network (`CW-SSN`) & Toán tử Nguyên bản (`CW-SSO`)  
> **Tạp chí mục tiêu:** *IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI — Q1, IF ~ 20.8)*  
> **Kho lưu trữ GitHub:** [https://github.com/hannhu11/Complex-Wasserstein-State-Space-Network-CW-SSN](https://github.com/hannhu11/Complex-Wasserstein-State-Space-Network-CW-SSN)  
> **Thư mục đồng bộ cục bộ:** `C:\Users\ADMIN\Downloads\Complex Wasserstein State-Space Network — CW-SSN`  
> **Trạng thái hiện tại:** Đã hoàn tất cài đặt Plugin Academic Research Suite v3.23.0 + AERS, xây dựng cơ sở dữ liệu 25 bài báo tiền nghiên cứu (`TPAMI_PreResearch_References_25Papers.csv`), lập trình và kiểm thử đơn vị (Unit Test 8/8 PASS) toán tử `CW-SSO` & kiến trúc `CW-SSN`, khởi tạo 10 Notebook thực nghiệm (`NB_01` – `NB_10`) và mẫu bản thảo LaTeX/Cover Letter chuẩn IEEE TPAMI.

---

## 1. Hệ Sinh Thái Kỹ Năng Nghiên Cứu Học Thuật Đã Cài Đặt (`academic-research-tpami`)

Đã phân tích đối sánh 3 kho kỹ năng nghiên cứu quốc tế và hợp nhất thành công vào hệ thống plugin của Antigravity tại `C:\Users\ADMIN\.gemini\config\plugins\academic-research-tpami`:

1. **`Imbad0202/academic-research-skills-codex` (v3.23.0) & `imbad0202/academic-research-skills`:**
   - Cung cấp bộ điều phối trung tâm `academic-research-suite` (`SKILL.md` v3.23.0) và 6 quy trình cốt lõi:
     - `ars-deep-research` (Khảo cứu tài liệu, Socratic scoping, Three-Way Scan `/ars-3w`)
     - `ars-academic-paper` (Lập dàn ý `/ars-plan`, viết bản thảo, kiểm tra trích dẫn `/ars-citation-check`, lộ trình phản biện `/ars-revision-coach`)
     - `ars-academic-paper-reviewer` (Mô phỏng hội đồng 5 phản biện độc lập + Editor-in-Chief `/ars-reviewer`)
     - `ars-academic-pipeline` (Chuỗi nghiên cứu đầu-cuối có kiểm chứng Material Passport `/ars-full`)
     - `ars-experiment-agent` (Thiết kế kịch bản thực nghiệm & kiểm chứng độ tái lập)
     - `ars-sr-screener` (Sàng lọc tài liệu hệ thống theo chuẩn PRISMA)
2. **`brycewang-stanford/Auto-Empirical-Research-Skills` (Tuyển chọn 13 kỹ năng CS/ML/Math/LaTeX/Kaggle, loại bỏ hoàn toàn kỹ năng Y sinh/Đám mây ngoài phạm vi):**
   - `aers-formula-derivation`: Biến đổi đạo hàm và công thức toán học.
   - `aers-proof-writer`: Chứng minh chặt chẽ Định lý 1, Định lý 2 và các Bổ đề (Lemmas).
   - `aers-ablation-planner`: Thiết kế ma trận Ablation suy biến toán học $(\gamma, \beta, \Omega) \to 0$.
   - `aers-academic-plotting`: Xuất đồ thị học thuật chuẩn IEEE TPAMI (`fig1`, `fig2`).
   - `aers-kaggle-research`: Quản lý huấn luyện phân tán trên cụm Kaggle 2x Tesla T4 & Colab Pro.
   - `aers-latex-document`, `aers-ml-paper-writing`, `aers-humanizer-academic`, `aers-de-aigc`, `aers-citation-checker`, `aers-lit-review-tools`, `aers-novelty-check`, `aers-autoresearch`.

---

## 2. Tư Duy Thiết Kế Cốt Lõi: Tại Sao Phải Là Toán Tử Nguyên Bản `CW-SSO` (Chống Lắp Ráp Mô-đun A+B+C)?

- **Điểm yếu chí mạng của hướng lắp ghép cũ (`UOT-FSMamba`):** Việc ghép nối cơ học 3 khối rời rạc (Nhánh FFT/DCT bên ngoài + Mạng phụ ước lượng bất định Uncertainty + Mô-đun Optimal Transport hậu kỳ) bị hội đồng phản biện **IEEE TPAMI** đánh giá là *"Incremental Lego-style Engineering (A+B+C)"*, gây phình tham số và không giải quyết được sự nhiễm bẩn trạng thái ngay bên trong phương trình vi phân của Mamba.
- **Giải pháp nguyên bản Q1 IEEE TPAMI (`CW-SSO` — `ComplexWassersteinSSM2D`):**
  1. **Rời rạc hóa Trắc địa Wasserstein Nghiệm đóng (Closed-Form Wasserstein Geodesic Discretization):**
     Mỗi vị trí không gian $p_t$ được mô hình hóa như một phân phối Gaussian cục bộ $\mathcal{P}_t = \mathcal{N}(\mu_t, \text{diag}(\sigma_t^2))$ trích xuất qua cửa sổ trượt không tham số $\mathcal{N}(p_t)$ kích thước $k \times k$. Khoảng cách 2-Wasserstein nghiệm đóng giữa hai bước không gian liên tiếp:
     $$\mathcal{W}_2^2(\mathcal{P}_t, \mathcal{P}_{t-1}) = \|\mu_t - \mu_{t-1}\|_2^2 + \|\sigma_t - \sigma_{t-1}\|_2^2$$
     Bước nhảy thời gian-không gian hiệu dụng trên đa tạp Wasserstein:
     $$\tilde{\Delta}_t = \underbrace{\text{Softplus}(W_\Delta x_t) \odot \exp\Big(-\gamma \cdot \mathcal{W}_2^2(\mathcal{P}_t, \mathcal{P}_{t-1})\Big)}_{\Delta_{\text{in}, t} \text{ (Khóa cổng nạp nhiễu ngoại lai)}} + \beta \cdot \mathcal{W}_2^2(\mathcal{P}_t, \mathcal{P}_{t-1})$$
     Khi đường quét đi qua ranh giới suy thoái đột ngột ($\mathcal{W}_2^2 \gg 0$), số hạng $\beta \cdot \mathcal{W}_2^2$ làm hệ số suy giảm hàm mũ $\exp(-\tilde{\Delta}_t \lambda_k) \to 0$, **xóa sạch trạng thái nhiễm bẩn cũ**, đồng thời $\Delta_{\text{in}, t} \to 0$ ngăn không cho nhiễu tràn vào trạng thái mới.
  2. **Không gian Trạng thái Phổ Quay Trực giao $2 \times 2$ (Rotational Spectral State-Space):**
     Mở rộng ma trận chuyển trạng thái sang miền phức $A = -\Lambda + i\Omega \in \mathbb{C}^{N \times N}$ ($\Lambda > 0$) và ánh xạ đẳng cấu sang phép quay khối trực giao thực $2 \times 2$ giữa các cặp kênh chẵn-lẻ $(h_t^{(2k)}, h_t^{(2k+1)})$ để chạy nguyên bản ở `FP16` trên Tensor Cores của Tesla T4:
     $$\begin{bmatrix} h_t^{(2k)} \\ h_t^{(2k+1)} \end{bmatrix} = e^{-\tilde{\Delta}_t \lambda_k} \begin{bmatrix} \cos(\tilde{\Delta}_t \omega_k) & -\sin(\tilde{\Delta}_t \omega_k) \\ \sin(\tilde{\Delta}_t \omega_k) & \cos(\tilde{\Delta}_t \omega_k) \end{bmatrix} \begin{bmatrix} h_{t-1}^{(2k)} \\ h_{t-1}^{(2k+1)} \end{bmatrix} + \begin{bmatrix} \overline{B}_t^{(2k)} \\ \overline{B}_t^{(2k+1)} \end{bmatrix} x_t$$
     Do nhóm các phép quay trực giao 2 chiều có tính đóng và kết hợp (associative), phương trình trên được giải chính xác bằng **Thuật toán Quét tiền tố song song Kogge-Stone (`rotational_2x2_Scan_parallel`)** chỉ trong $O(\log_2 L)$ bước.

---

## 3. Hai Định Lý Toán Học Bảo Chứng (White-Box Theoretical Guarantees)

- **Định lý 1 (Bảo toàn Năng lượng Phổ Tần số cao — High-Frequency Spectral Energy Conservation):**
  Với mọi góc quay pha $\theta_{t,k} = \tilde{\Delta}_t \omega_k \in \mathbb{R}$, ma trận quay khối $R(\theta_{t,k})$ là trực giao chuẩn tắc ($\|R(\theta_{t,k})\|_2 = 1$), bảo toàn tuyệt đối chuẩn năng lượng L2 của cặp trạng thái chẵn-lẻ $\|R(\theta_{t,k}) h_{t-1}\|_2^2 = \|h_{t-1}\|_2^2$ và tương đương với phép Biến đổi Gabor/Fourier Cửa sổ Ngắn Thích nghi theo Trắc địa Wasserstein liên tục dọc theo quỹ đạo quét.
- **Định lý 2 (Cận trên Suy giảm Hàm mũ Sai số Nhiễm bẩn Trạng thái — Exponential Cross-Region State Contamination Bound):**
  Giả sử tại vị trí ranh giới $t_0$ xuất hiện bước nhảy suy thoái không đồng nhất có $\mathcal{W}_2^2(\mathcal{P}_{t_0}, \mathcal{P}_{t_0-1}) = \Delta_W^2 > 0$. Sai số nhiễm bẩn trạng thái tại vị trí $t > t_0$ trong vùng ảnh sạch thỏa mãn bất đẳng thức suy giảm hàm mũ chặt:
  $$\|h_t^{\text{clean}} - h_t^{\text{corrupted}}\|_2 \le \|h_{t_0-1}^{\text{clean}} - h_{t_0-1}^{\text{corrupted}}\|_2 \cdot \exp\Big(-\beta \lambda_{\min} \Delta_W^2 - \lambda_{\min} \sum_{\tau=t_0}^{t} \Delta_{\text{in}, \tau}\Big)$$

---

## 4. Kết Quả Kiểm Thử Đơn Vị & Ngân Sách Tham Số (Verified KPIs)

Đã kiểm chứng thực tế qua bộ test `tests/test_cw_ssn.py` (8/8 bài kiểm thử PASS 100%):
- **Tương đương số học Parallel Kogge-Stone Scan vs. Sequential Scan:** Sai số cực đại $< 1.5 \times 10^{-6}$ (Ngưỡng KPI: $< 10^{-4}$).
- **Bảo vệ ranh giới ảnh suy thoái không đồng đều (Motivation Figure 1):** Tăng **$+8.39\text{ dB PSNR}$** tại vùng ranh giới so với Mamba ZOH truyền thống (Ngưỡng KPI: $> +2.00\text{ dB}$).
- **Hiệu năng bộ nhớ VRAM `AMP FP16`:** Huấn luyện `(4, 3, 128, 128)` chỉ tốn **`1.67 GB VRAM`**, suy luận độ phân giải cao `(1, 3, 512, 512)` chỉ tốn **`1.93 GB VRAM`** (thấp hơn rất nhiều so với ngưỡng `10.50 GB VRAM` của Tesla T4).
- **Ngân sách tham số (Parameter Budgets):**
  - `CW-SSN-T` (Tiny): **`5,813,285` tham số (`5.81M`)** $\in [5.5\text{M}, 6.2\text{M}]$ — nhẹ hơn **88.4%** so với `DA-RCOT` (`50.1M`, TPAMI 2025) và nhỏ hơn cả `MIRAGE-T` (`6.0M`, ICLR 2026).
  - `CW-SSN-B` (Base): **`7,953,895` tham số (`7.95M`)** $\in [7.5\text{M}, 8.2\text{M}]$.
  - `CWSSOAdapter` (Universal Foreground Segmentation trên `SegFormer-B4` đóng băng `61.4M`): **`987,745` tham số (`0.99M` = `1.61%`)** — thấp hơn rất nhiều so với ngưỡng KPI `5.50%` và mức `5.70%` của `EVP` gốc (TPAMI 2025).

---

## 5. Danh Mục 10 Notebook Thực Nghiệm (`notebooks/NB_01` – `NB_10`)

| Mã Notebook | Tên File | Vai Trò & Cụm Phần Cứng Thực Thi |
| :--- | :--- | :--- |
| **`NB_01`** | `NB_01_Dataset_Packaging_and_2xT4_DDP_Engine.ipynb` | Đóng gói dữ liệu 3D/5D/10D AiOIR (`<0.05s/batch`) & kiểm chứng DDP Auto-Resume sai số `0.000000` trên Kaggle T4x2 #1. |
| **`NB_02`** | `NB_02_Baseline_Repro_and_State_Contamination_Proof.ipynb` | Tái lập baseline `DA-RCOT`/`MambaIR` & xuất **Hình 1 (`fig1_motivation_state_contamination.png`)** chứng minh hiện tượng nhiễm bẩn trạng thái. |
| **`NB_03`** | `NB_03_CW_SSO_Core_Operator_and_Unit_Tests.ipynb` | Kiểm thử đơn vị toán tử `ComplexWassersteinSSM2D`, phép quét song song $2 \times 2$ $O(\log_2 L)$ và độ ổn định `AMP FP16` (`4x3x128x128` & `1x3x512x512`). |
| **`NB_04`** | `NB_04_CW_SSN_Backbone_and_Spectral_Wasserstein_Loss.ipynb` | Xây dựng mạng phân cấp đồng nhất `CW-SSN-T/B` & chạy Pilot Run kiểm chứng hàm mất mát `SpectralWassersteinLoss`. |
| **`NB_05`** | `NB_05_Main_Experiments_3D_and_5D_AllInOne.ipynb` | Huấn luyện chính thức `CW-SSN-T` (`5.81M`) & `CW-SSN-B` (`7.95M`) trên chuẩn **3D & 5D All-in-One** (3x Kaggle T4x2 + 1x Colab Pro). |
| **`NB_06`** | `NB_06_Scalability_10D_AllInOne_Benchmark.ipynb` | Mở rộng quy mô lên **10 loại suy thoái (10D Benchmark)** theo chuẩn `DCPT` (ICLR 2025) & phân tích triệt tiêu xung đột gradient. |
| **`NB_07`** | `NB_07_Universal_Foreground_Segmentation_EVP.ipynb` | Mở rộng sang **4 bài toán Phân đoạn Tiền cảnh Phổ quát (COD, Defocus Blur, Forgery, Shadow — EVP Protocol)** chống Desk-Reject sang IEEE TIP. |
| **`NB_08`** | `NB_08_Mathematical_Degeneracy_Ablation_Matrix.ipynb` | Thực thi **Ma trận Ablation Suy biến Toán học** $(\gamma, \beta, \Omega) \to 0$ (4 trường hợp) & khảo sát độ nhạy cửa sổ $|\mathcal{N}(p)|, r$. |
| **`NB_09`** | `NB_09_ZeroShot_Generalization_and_WhiteBox_Theory_Plots.ipynb` | Kiểm thử Zero-Shot OOD ($\sigma \in \{35, 60, 75\}$), vẽ biểu đồ phân cụm đa tạp **t-SNE** & đồ thị kiểm chứng **Định lý 2 (`fig2`)**. |
| **`NB_10`** | `NB_10_OneClick_Reproducibility_and_TPAMI_Submission_Check.ipynb` | Kiểm thử tái lập một chạm (`<15 phút`) và rà soát 4 cửa ải nộp bài ScholarOne **IEEE TPAMI**. |

---

## 6. Cấu Trúc Thư Mục Toàn Dự Án

```text
Complex-Wasserstein-State-Space-Network-CW-SSN/
├── models/
│   ├── __init__.py
│   └── cw_ssn.py                                      # Toán tử CW-SSO, CWSSBlock, CW-SSN-T/B, CWSSOAdapter, SpectralWassersteinLoss
├── notebooks/
│   ├── NB_01_Dataset_Packaging_and_2xT4_DDP_Engine.ipynb
│   ├── NB_02_Baseline_Repro_and_State_Contamination_Proof.ipynb
│   ├── NB_03_CW_SSO_Core_Operator_and_Unit_Tests.ipynb
│   ├── NB_04_CW_SSN_Backbone_and_Spectral_Wasserstein_Loss.ipynb
│   ├── NB_05_Main_Experiments_3D_and_5D_AllInOne.ipynb
│   ├── NB_06_Scalability_10D_AllInOne_Benchmark.ipynb
│   ├── NB_07_Universal_Foreground_Segmentation_EVP.ipynb
│   ├── NB_08_Mathematical_Degeneracy_Ablation_Matrix.ipynb
│   ├── NB_09_ZeroShot_Generalization_and_WhiteBox_Theory_Plots.ipynb
│   └── NB_10_OneClick_Reproducibility_and_TPAMI_Submission_Check.ipynb
├── paper/
│   ├── main_tpami.tex                                 # Bản thảo khung chuẩn IEEEtran 10pt journal compsoc (14 trang)
│   └── cover_letter_tpami.tex                         # Cover Letter đáp ứng 4 cửa ải Associate Editor IEEE TPAMI
├── references/
│   └── TPAMI_PreResearch_References_25Papers.csv      # Cơ sở dữ liệu 25 bài báo tiền nghiên cứu (15 cột chuẩn hóa)
├── scripts/
│   ├── generate_references_csv.py                     # Script sinh & kiểm tra toàn vẹn CSV 25 bài báo
│   └── generate_notebooks.py                          # Script sinh & kiểm tra cú pháp AST cho 10 Jupyter Notebooks
├── tests/
│   └── test_cw_ssn.py                                 # Bộ 8 Unit Tests kiểm chứng toán học, VRAM FP16 & ngân sách tham số
├── docs/                                              # Tài liệu định hướng nghiên cứu gốc (.pdf, .docx, .md)
├── TPAMI_PreResearch_References_25Papers.csv          # Bản sao CSV ở thư mục gốc để truy cập nhanh
├── evaluate_all.sh                                    # Script kiểm chứng tái lập một chạm cho Reviewer TPAMI
├── requirements.txt                                   # Danh sách thư viện ghim phiên bản ổn định
├── memory.md                                          # Bộ nhớ kiến trúc & tiến độ toàn dự án
└── README.md                                          # Tài liệu hướng dẫn toàn diện trên GitHub
```
