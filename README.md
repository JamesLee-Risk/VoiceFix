# VoiceFix — 音聲／歌聲清理工具

用一步一步的引導流程，讓不懂聲音工程的人也能把原始人聲錄音處理乾淨。提供三條流程：**流程 A 音聲作品**（ASMR、廣播劇、雙耳錄音）、**流程 B 歌聲**（翻唱、原創歌曲的主唱）與**流程 C 多軌歌聲**（1 條主唱＋最多 4 條和聲／疊唱）。

## 下載（Windows）

到 GitHub 的 Releases 下載 `VoiceFix-<版本>-windows-x64.zip`，解壓縮後執行資料夾裡的 `VoiceFix.exe`，不需要安裝 Python。

- 打包版不含選用的 AI 降噪（PyTorch 約 1 GB）；需要的話請用下方的原始碼方式安裝。
- 遇到問題時，記錄檔在 `%LOCALAPPDATA%\VoiceFixoicefix.log`。
- 檢查安裝是否完整：在命令提示字元執行 `VoiceFix.exe --self-test 結果.txt`，再打開 `結果.txt`。

## 從原始碼安裝與執行

需要 Python 3.10 以上（Windows 10/11）。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m voicefix          # 開啟程式
.venv\Scripts\python -m pytest -q         # 執行測試
```

選用：AI 降噪（DeepFilterNet，連同 PyTorch 約 1 GB）。裝好後 A4／B3 的「處理方式」會多出「AI 降噪」：

```powershell
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[ai]"
```

第一次使用時會下載模型（只下載模型，音訊一律在本機處理）。AI 降噪是用說話聲訓練的，可能把舔耳、口腔音等表演音或歌聲當成噪音，所以不是預設值；選用時畫面會提醒用「被移除的聲音」確認。

流程 C（多軌）：開新專案時先選主唱，再一次多選和聲／疊唱（最多 4 條，需從同一個錄音專案匯出、起點對齊、取樣率相同）。畫面上方的「目前的軌」切換要檢查的軌；每一步的設定預設全部軌共用，勾「此軌單獨設定」可個別調整；按「套用」會一起算完所有軌。

操作：波形上拖曳＝選取（Ctrl＋拖曳＝多選）、點一下＝移動播放位置、滾輪＝縮放、右鍵拖曳＝平移。
空白鍵＝播放／停止，B＝切換處理前／處理後，R＝只聽被移除的聲音。

自己打包 Windows 版：`.venv\Scripts\python -m pip install pyinstaller`，再執行 `.venv\Scripts\python packaging/build.py`。它會打包、跑自我檢查，通過後在 `dist/` 產生 zip。

## 目前進度（MVP）

| 範圍 | 狀態 |
| --- | --- |
| 共通：專案檔、自動儲存、步驟狀態（未開始／已套用／已略過／需重算） | 完成 |
| 共通：處理前／處理後／被移除的聲音試聽、15 ms 交叉淡化、LUFS 音量對齊 | 完成 |
| 共通：三種套用範圍、事件清單逐筆勾選、保護區段、邊界 8 ms 淡入淡出 | 完成 |
| 共通：只算試聽範圍的預覽、背景整檔算圖（可取消）、步驟輸出快取 | 完成 |
| 流程 A MVP 步驟：A0、A1、A2、A3、A4、A5、A6、A8、A10、A11 | 完成 |
| 流程 A v1 步驟：A7 呼吸聲（預設 −3 dB）、A9 補底噪（用錄音自己的背景聲） | 完成 |
| 流程 B MVP 步驟：B0、B2、B3、B4、B5、B8、B9、B10、B12 | 完成 |
| 流程 B：伴奏一起播（B8～B12 預設勾選）、乾聲（不含 B11）／混合試聽版匯出 | 完成 |
| 流程 B v1 步驟：B6 音準修正（自然／明顯效果）、B11 空間感 | 完成 |
| B6／C6 逐音指定目標音（曲線標出目標音名、點選音後 ↑↓ 半音） | 完成 |
| 主唱對齊伴奏拍點：B7、C7a（拍點偵測、搶拍／拖拍逐筆清單、保持音高的時間微調） | 完成 |
| 流程 C 多軌歌聲：C0～C12（含 C7b 和聲對齊主唱）、參數共用可個別微調、分軌與混合試聽版匯出 | 完成 |
| B1 選段拼接（同一聲部的多個 take） | 暫緩（見待決事項） |
| 介面多語系（英文、日文） | 未開始；字串目前直接寫在程式中 |

## 架構

```
voicefix/
  core/      專案模型、步驟鏈（快取、需重算判斷）、分塊處理、區段遮罩、時間軸對應、響度、匯出
  dsp/       濾波器、降噪、口水音／爆破音、換氣、齒音、壓縮、限制器、音高偵測與移調
  steps/     每個步驟實作同一個 Step 介面：analyze / map_strength / measure / prepare / process
             （流程 C 沿用流程 B 的步驟，另有 C7b 和聲對齊主唱與以主唱為準的 C12）
  playback/  A/B 試聽混音器（純邏輯、可測試）與音效卡輸出
  ui/        PySide6 介面：步驟清單、波形與頻譜、控制欄、試聽列
```

設計重點：

- **輸出是純函數**：步驟輸出只由「原始檔＋各步參數」決定，快取以內容雜湊為鍵，所以前面沒變就不重算，重開專案匯出結果逐樣本相同。
- **區段一律以原始檔時間儲存**：A1 刪除片段後，選取、保護區段、事件勾選都會自動對應到新時間軸。
- **分塊處理**：所有步驟以固定 30 秒格點加前後緩衝處理，寫入磁碟快取（.npy，記憶體映射讀取），長檔不會吃光記憶體；預覽與整檔算圖用同一套程式，聽到的就是會匯出的。
- **保護區段與套用範圍由步驟鏈統一處理**：步驟只管 DSP，原樣保留的區域逐樣本不變。
- **多軌（流程 C）**：每軌是一個完整的單軌專案與步驟鏈；共用參數、步驟狀態由上層同步。和聲的 C7b／C12 會讀取主唱在該步的輸入，並把它納入快取鍵，主唱改了和聲就會自動重算。
- **左右聲道連動**：所有增益與遮罩由兩聲道共同計算、零相位濾波，不會產生左右時間差。

## 待決事項

- A10「安靜／標準」目標響度暫定 −24／−18 LUFS（`voicefix/steps/flow_a/level_export.py` 的 `LOUDNESS_PRESETS`），需量測已上架作品後決定。
- B12「交給混音師／自己使用」目標響度暫定 −20／−16 LUFS（`voicefix/steps/flow_b/song.py` 的 `VOCAL_LOUDNESS_PRESETS`）。
- 同一聲部的多個 take 選段拼接（B1）：暫緩，待決定如何簡單呈現後再做。多聲部（主唱＋和聲）改由流程 C 處理。
- B6 自然模式的修正門檻為偏離 30 音分（PRD 的初始建議值，待測試調整；可在「進階」修改）。偏差超過半音的一半時，程式會把它當成隔壁的音；這種音要在音高曲線上點選後手動指定目標音。
- B8 只自動處理 1.5～6 kHz 的明顯窄共振；中低頻的「混濁」無法從長期頻譜和歌手的基頻區分，所以交給風格預設處理。
- 降噪品質仍是最大技術風險：AI 降噪（DeepFilterNet）以語音訓練，對耳語、舔耳等表演音與歌聲的效果需實測。

## 授權

VoiceFix 是自由軟體，以 **GNU General Public License v3.0 或更新版本（GPL-3.0-or-later）** 釋出，全文見 [LICENSE](LICENSE)。
選擇 GPL 是為了能使用只允許開源軟體使用的套件：B6 音準修正用的 Rubber Band 與 B11 空間感用的殘響都來自 pedalboard（GPL-3.0）。

使用的第三方套件：

| 套件 | 用途 | 授權 | 安裝 |
| --- | --- | --- | --- |
| PySide6（Qt） | 介面 | LGPL-3.0 | 必要 |
| pyqtgraph | 波形與頻譜 | MIT | 必要 |
| numpy、scipy | 數值運算、濾波、FFT | BSD-3-Clause | 必要 |
| soundfile（libsndfile） | 讀寫音檔 | BSD-3-Clause（libsndfile 為 LGPL-2.1+） | 必要 |
| soxr（libsoxr） | 伴奏重取樣 | LGPL-2.1+ | 必要 |
| pedalboard（含 Rubber Band、JUCE） | B6 自然模式移調（保留共振峰）、B11 殘響 | GPL-3.0 | 必要 |
| pyworld（WORLD） | B6 音高偵測 | MIT（WORLD 為 modified BSD） | 必要 |
| sounddevice（PortAudio） | 播放 | MIT | 必要 |
| DeepFilterNet、PyTorch | AI 降噪（選用） | MIT／Apache-2.0、BSD-3-Clause | `pip install -e ".[ai]"` |
| pytest、pytest-cov、pyloudnorm | 測試（pyloudnorm 只用來驗證響度量測） | MIT | `pip install -e ".[dev]"` |

降噪、口水音／噴麥修補、EQ、壓縮、齒音、限制器、響度量測、補底噪與 B6 明顯效果用的 TD-PSOLA 都是本專案自行實作（`voicefix/dsp/`）。這不是授權考量：這些實作是零相位、分塊結果與整檔一致，才能保證試聽聽到的就是匯出的結果。

