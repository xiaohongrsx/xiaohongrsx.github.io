# xiaohongrsx.github.io

renshixin's blog — 基于 [Typst](https://typst.app/) 的静态博客，使用自定义 Node.js 构建脚本生成 HTML。

## 环境要求

- [Node.js](https://nodejs.org/) >= 20
- [Typst](https://github.com/typst/typst) >= 0.14

## 本地开发

```bash
# 构建网站（4 并发）
npm run build:fast

# 本地预览（端口 11451）
npm run serve
```

浏览器打开 `http://localhost:11451` 即可预览。

## 其他构建命令

| 命令 | 说明 |
|------|------|
| `npm run build` | 单线程构建 |
| `npm run build:fast` | 4 并发构建 |
| `npm run build:preview` | 构建到 `_site-preview` |
| `npm run build:preview_fast` | 4 并发构建到 `_site-preview` |

CLI 选项：`-o` 输出目录、`-f` 强制全量构建、`-j` 并发数、`-h` 帮助。

## 目录结构

```
├── posts/          # 博客文章（每篇一个目录，含 index.typ）
├── pages/          # 独立页面（首页、关于、归档、标签、分类）
├── assets/         # 静态资源（CSS、JS、图标）
├── lib/            # 构建脚本与 Typst 模板库
├── config.typ      # 主题与导航配置
├── site.config.json# 站点元数据（RSS、标题、作者）
└── _site/          # 构建产物（已 gitignore）
```

## 写文章

在 `posts/` 下新建目录，创建 `index.typ`，参考已有文章的格式编写即可。

## 部署

推送到 `main` 分支后，GitHub Actions 自动构建并部署到 GitHub Pages。

## License

[MIT](LICENSE)
