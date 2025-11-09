# Git Bash/Windows 代理配置指南（HTTP/HTTPS 与 SSH）

本指南整理在 Windows 的 Git Bash 环境下，为各类命令行程序（通用、Git HTTP/HTTPS、Git SSH）配置代理的常用方法，并给出测试与排错建议。你的本地已确认存在 `connect`（/mingw64/bin/connect），因此 SSH 代理推荐使用 `connect`。

## 一、通用环境变量（适用于多数 CLI）
- 临时为单条命令设置代理（仅本次命令生效）：
  - `HTTPS_PROXY=http://127.0.0.1:1235 HTTP_PROXY=http://127.0.0.1:1235 <your_command>`
  - 也可使用大写/小写混用：多数工具同时识别 `HTTP_PROXY`/`HTTPS_PROXY` 与 `http_proxy`/`https_proxy`。
- 持久设置（Git Bash）：在 `~/.bashrc` 追加：
  ```bash
  export HTTP_PROXY=http://127.0.0.1:1235
  export HTTPS_PROXY=http://127.0.0.1:1235
  export NO_PROXY=localhost,127.0.0.1,::1
  ```
  然后 `source ~/.bashrc` 使其生效。
- SOCKS5 示例：`HTTP_PROXY=socks5://127.0.0.1:1080`，`HTTPS_PROXY=socks5://127.0.0.1:1080`。
- 含认证信息示例：`http://user:pass@127.0.0.1:1235`（如密码含特殊字符，需 URL 编码）。

常见工具的额外配置：
- pip（Windows，用户级配置）：在 `%APPDATA%\pip\pip.ini` 写入：
  ```ini
  [global]
  proxy = http://127.0.0.1:1235
  ```
  或临时：`pip install <pkg> --proxy http://127.0.0.1:1235`。
- npm：`npm config set proxy http://127.0.0.1:1235 && npm config set https-proxy http://127.0.0.1:1235`。
- conda：
  ```bash
  conda config --set proxy_servers.http http://127.0.0.1:1235
  conda config --set proxy_servers.https http://127.0.0.1:1235
  ```

## 二、为 Git 配置 HTTP/HTTPS 代理
> 说明：Git 的环境变量支持在部分环境下不稳定，建议优先使用 `git config`。

- 全局代理（所有 HTTP/HTTPS 仓库生效）：
  ```bash
  git config --global http.proxy  http://127.0.0.1:1235
  git config --global https.proxy http://127.0.0.1:1235
  ```
- 针对特定域名（例如仅 github.com）：
  ```bash
  git config --global http.https://github.com.proxy  http://127.0.0.1:1235
  git config --global https.https://github.com.proxy http://127.0.0.1:1235
  ```
  注意：以上语法中 `http.https://github.com.proxy` 为一整个键名（中间不应有空格）。
- 仅在当前仓库生效（在仓库根目录执行）：
  ```bash
  git config http.proxy  http://127.0.0.1:1235
  git config https.proxy http://127.0.0.1:1235
  ```
- SOCKS5 示例：
  ```bash
  git config --global http.proxy  socks5://127.0.0.1:1080
  git config --global https.proxy socks5://127.0.0.1:1080
  ```
- 验证与清理：
  ```bash
  git config --global --get http.proxy
  git config --global --get https.proxy
  git config --global --unset http.proxy
  git config --global --unset https.proxy
  ```

## 三、为 Git 配置 SSH 代理（使用 connect）
> 你的环境已存在 `/mingw64/bin/connect`，适合用作 `ProxyCommand`。`nc` 不存在，因此不要使用 `nc -X connect -x ...`。

- 在 `~/.ssh/config` 添加针对 `github.com` 的代理：
  ```sshconfig
  Host github.com
      ProxyCommand connect -H 127.0.0.1:1235 %h %p
  ```
  - `-H` 表示通过 HTTP 代理；若为 SOCKS5，使用：
    ```sshconfig
    Host github.com
        ProxyCommand connect -S 127.0.0.1:1080 %h %p
    ```
- 临时（不改 config）：
  ```bash
  GIT_SSH_COMMAND='ssh -o ProxyCommand="connect -H 127.0.0.1:1235 %h %p"' \
    git ls-remote git@github.com:github/hub.git
  ```
- 测试 SSH 连接：
  ```bash
  ssh -T git@github.com
  ```
  可能输出：
  - `Hi <user>! You've successfully authenticated...`（已配置 SSH key 并代理可用）。
  - `Permission denied (publickey).`（代理可用但未配置 SSH key）。

## 四、常见排错
- 代理端口不通：确认本地代理监听端口（如 1235/1080）正确、允许来自 Git Bash 的连接。
- 证书问题：企业代理可能需要根证书；Git 报错可尝试设置 `git config --global http.sslVerify false`（不建议长期关闭）。
- 凭证包含特殊字符：在 URL 中进行编码（例如 `@`、`:`、`#` 等）。
- 与 VPN/防火墙冲突：确认 VPN Split Tunneling 与防火墙策略允许代理进程访问外部网络。

## 五、与 ARPO 项目相关的建议
- 本项目包含从 Hugging Face 等外部源下载与安装依赖的步骤；若网络受限，建议在执行 `pip`, `git clone`, `wget/curl`, `npm` 等操作时，提前设置上述代理。
- 可在训练脚本前添加：
  ```bash
  export HTTP_PROXY=http://127.0.0.1:1235
  export HTTPS_PROXY=http://127.0.0.1:1235
  ```
  或在需要的命令前临时添加环境变量前缀。

---
如需我自动为当前用户写入 `~/.bashrc` 与 `~/.ssh/config` 的代理配置，并做连接测试（HTTP 拉取与 SSH 验证），请告诉我你的代理类型（HTTP 或 SOCKS5）与端口。我可以一键完成并回传测试结果。
