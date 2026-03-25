#import "../config.typ": *
#let posts = query-posts()

#let tags = (:)
#let categories = (:)

#for post in posts [
  #for tag in post.tags [
    #tags.insert(tag, true)
  ]

  #if post.category != "" [
    #categories.insert(post.category, true)
  ]
]

#show: template-page.with(
  title: "首页",
  description: "站点首页",
)

#{
  html.div(class: "homepage-header", {
    html.div(class: "homepage-header-carbon", "Xiao")
    html.div(class: "homepage-header-typst", "Hong")
    html.div(class: "homepage-header-blog", "RenShixin")
  })
}

= 关于

这是 Typst Blog 的基础信息页。

- 文章数量：#posts.len()
- 标签数量：#tags.keys().len()
- 分类数量：#categories.keys().len()

常用入口：
- #link("/posts/")[文章]
- #link("/tags/")[标签]
- #link("/categories/")[分类]
- #link("/archive/")[归档]

外部工具与服务：

- #link("https://typst.app/")[Typst]：用于文章与页面的排版及 HTML 导出。
- #link("https://nodejs.org/")[Node.js]：用于执行静态站点构建脚本（路由、元数据与页面生成）。
- #link("https://shiki.matsu.io/")[Shiki]：用于代码高亮渲染（通过 #link("https://esm.sh/")[esm.sh] 在线模块加载）。
- #link("https://www.jsdelivr.com/")[jsDelivr]：用于加载 IBM Plex 字体资源。
- #link("https://carbondesignsystem.com/")[Carbon Design System]：用于界面设计语言参考。

许可与版权说明：

本模板在视觉与实现上参考了以下公开项目与资源：

- Carbon Design System（界面设计系统）
- IBM Plex Mono（字体）
- IBM Plex Sans SC（字体）
- #link("https://github.com/Yousa-Mirage/Tufted-Blog-Template")[Tufted Blog Template]

相关名称、设计系统与字体的版权归其各自权利人所有；
本项目仅在其许可条款允许范围内进行参考与使用。
