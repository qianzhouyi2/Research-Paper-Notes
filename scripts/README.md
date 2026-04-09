# Scripts README

这个目录放的是站点构建与发布脚本，不属于 wiki 正文内容。

## 主要脚本

- `build_pages_site.py`：把仓库内 Markdown 构建成 `docs/` 静态站点（GitHub Pages 发布目录）。
- `prepare_github_wiki.py`：把内容整理成 GitHub Wiki 兼容输出。
- `publish.ps1`：Windows/PowerShell 一键发布流程。
- `publish.sh`：Bash 一键发布流程（WSL/Git Bash/Linux/macOS）。

## 一键发布（PowerShell）

在仓库根目录执行：

```powershell
.\scripts\publish.ps1 -Message "Update wiki and rebuild Pages"
```

可选参数：

- `-Message`：Git 提交信息（默认：`Update wiki and rebuild Pages`）
- `-SkipPagesCheck`：跳过 `gh api` 的 Pages 状态查询

## 一键发布（Bash）

在仓库根目录执行：

```bash
bash ./scripts/publish.sh "Update wiki and rebuild Pages"
```

如果不传参数，会使用默认提交信息。

## 脚本默认行为

`publish.ps1` / `publish.sh` 会自动执行：

1. 运行 `python scripts/build_pages_site.py`
2. 清理 `.obsidian` 运行态文件（不入库）
3. `git add --all -- ':/' ':(exclude).obsidian'`
4. 如果有变更则 `commit + push origin main`
5. 若本机安装了 `gh`，查询 Pages 最新构建状态

## 依赖

- Python 3（用于构建页面）
- Git
- `gh`（可选，仅用于查看 Pages 状态）
