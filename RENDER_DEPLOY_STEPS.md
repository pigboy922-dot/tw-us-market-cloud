# Render 部署步驟

這包是給 Render 用的雲端版：

```text
D:\V1046_FULL_MARKET_STRATEGY_DATA_PACK\CLOUD_DAILY_MARKET_POOL_CLOUD_READY
```

## 1. 上傳到 GitHub

只上傳 `CLOUD_DAILY_MARKET_POOL_CLOUD_READY` 這個資料夾內容，不要上傳整個 1.46GB 研究資料包。

Render 需要看到 repo 根目錄有：

```text
app.py
engine.py
build_dashboard.py
render.yaml
requirements.txt
data_live/
frozen_rules/
runtime_outputs/
state/
tools/
```

## 2. Render 建立 Blueprint

Render Dashboard 內選：

```text
New -> Blueprint
```

連到你的 GitHub repo，Render 會讀 `render.yaml`。

目前設定：

```text
service: daily-market-pool-cloud
region: singapore
plan: standard
disk: /var/data, 5GB
```

不要用 Free 跑正式版，Free 會睡眠，而且沒有 persistent disk。

## 3. 部署完成後

Render 會給你網址，例如：

```text
https://daily-market-pool-cloud.onrender.com/
```

手機和平板就開這個網址。

## 4. 每天怎麼用

首頁看：

- 台股持股
- 美股持股
- 推薦日期
- 買入區間
- 是否可買
- 換倉倒數

要更新市場時，按網頁上的更新市場按鈕。

背景會跑：

```text
tools/update_live_prices.py
engine.run_update()
refresh_target_current_prices()
build_dashboard.main()
export_excel_record()
```

這等同 Render 版的 `99_FULL_MARKET_UPDATE_AND_START`，不是跑 Windows CMD。

## 5. 看更新進度

直接開：

```text
/api/cloud-update/status
```

例如：

```text
https://daily-market-pool-cloud.onrender.com/api/cloud-update/status
```

## 6. 重要限制

Render 版不會直接執行 `.cmd`，因為 Render 是 Linux 環境。

Render 版做法是用 `app.py` 的 API 在背景跑同等 Python 流程。

正式使用請用 paid web service + persistent disk。否則更新後的資料可能在重啟或 redeploy 後消失。
