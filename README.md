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

## 评论系统（Giscus）

本博客在每篇文章页底部使用 [Giscus](https://giscus.app/zh-CN)，把评论以 GitHub Discussions 的形式存在一个 public 仓库里。相比 Gitalk，Giscus 走 GitHub App 授权，**不再需要任何 OAuth Client Secret，全部配置项都是公开值**。一次性配置四步即可启用：

1. **开启 Discussions**：在评论仓库（默认 `xiaohongrsx/blog-comments`）的 `Settings → General → Features` 里勾上 `Discussions`。
2. **新建评论分类**：进入该仓库 `Discussions` 标签页，右上角设置图标 → `New category`，名字填 `Comments`，类型选 `Announcement`（只允许维护者发起话题，可有效防止灌水）。
3. **安装 Giscus App**：访问 [github.com/apps/giscus](https://github.com/apps/giscus) 把 Giscus 安装到评论仓库（限定到这个仓库即可，无需授权所有仓库）。
4. **抓取 `repoId` / `categoryId`**：访问 [giscus.app/zh-CN](https://giscus.app/zh-CN) 配置生成器，仓库填 `xiaohongrsx/blog-comments`，Mapping 选 `Pathname`，Discussion Category 选 `Comments`；页面下方生成的 `<script>` 标签里复制 `data-repo-id` 与 `data-category-id` 的值，填到 `site.config.json`：

   ```json
   {
     "giscus": {
       "repo": "xiaohongrsx/blog-comments",
       "repoId": "<从 giscus.app 抓到的 repoId>",
       "category": "Comments",
       "categoryId": "<从 giscus.app 抓到的 categoryId>",
       "mapping": "pathname",
       "strict": "0",
       "reactionsEnabled": "1",
       "emitMetadata": "0",
       "inputPosition": "bottom",
       "lang": "zh-CN"
     }
   }
   ```

完成后重新跑一次 `npm run build:fast`，构建脚本会把这份配置写到 `_site/assets/core/giscus.config.js`，文章页加载时会自动初始化评论组件。`repoId` 或 `categoryId` 为空时，前端只会在控制台打 `warn`，不会渲染评论框，方便本地调试。

> 备注：`repoId` / `categoryId` 是 GitHub GraphQL 暴露的对象 ID，本身就是公开可枚举的值，不会触发 GitHub Secret Scanning 告警，也不存在密钥泄露问题。主题切换通过监听 `<html>` 的 `data-theme` 属性走 `postMessage` 实时同步给 Giscus iframe，无需重载页面。

## 部署

推送到 `main` 分支后，GitHub Actions 自动构建并部署到 GitHub Pages。

## License

[MIT](LICENSE)
