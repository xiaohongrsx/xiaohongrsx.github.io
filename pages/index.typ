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
  title: "Home",
  description: "Ren Shixin's Personal Homepage",
)

#{
  html.div(class: "homepage-header", {
    html.div(class: "homepage-header-carbon", "Xiao")
    html.div(class: "homepage-header-typst", "Hong")
    html.div(class: "homepage-header-blog", "RenShixin")
  })
}

#{
  html.div(class: "about-section", {
    html.div(class: "about-photo", {
      html.elem("img", attrs: (src: "/assets/fig/rsx.jpg", alt: "Ren Shixin", class: "avatar"))[]
    })
    html.div(class: "about-info", {
      html.div(class: "about-name", "Ren Shixin")
      html.div(class: "about-name-en", "任世鑫")
      html.div(class: "about-affiliation", {
        html.elem("strong")[Tsinghua University]
        [, Zhili College, Information and Computational Science | Undergraduate]
      })
      html.div(class: "about-intro", [
        Hi! I am an undergraduate student (Class of 2023) majoring in Information and Computational Science at Zhili College, Tsinghua University. I am currently doing research at the Natural Language Processing Lab (THUNLP), Tsinghua University. My research interests include LLM Algorithms, LLM Systems, and High-Performance Computing (HPC).
      ])
      html.div(class: "contact-info", {
        html.div(class: "contact-item", {
          html.elem("img", attrs: (src: "/assets/icons/fa-envelope.svg", alt: "Email", class: "contact-icon"))[]
          html.a(href: "mailto:rsx23@mails.tsinghua.edu.cn", "rsx23@mails.tsinghua.edu.cn")
        })
        html.div(class: "contact-item", {
          html.elem("img", attrs: (src: "/assets/icons/fa-github.svg", alt: "GitHub", class: "contact-icon"))[]
          html.a(href: "https://github.com/xiaohongrsx", "xiaohongrsx")
        })
        html.div(class: "contact-item", {
          html.elem("img", attrs: (src: "/assets/icons/fa-weixin.svg", alt: "WeChat", class: "contact-icon"))[]
          html.span("xiaohongrsx")
        })
      })
    })
  })
}

= Education

#{
  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "Tsinghua University · Zhili College")
      html.div(class: "entry-time", "2023.09 -- Present")
    })
    html.div(class: "entry-meta", "Undergraduate | Information and Computational Science")
  })

  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "Nankai High School, Tianjin")
      html.div(class: "entry-time", "2020.09 -- 2023.08")
    })
  })
}

= Projects

#{
  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "Chinese Pre-training Data Synthesis via Web Rewriting")
      html.div(class: "entry-time", "2025.10 -- Present")
    })
    html.div(class: "entry-meta", "Main Contributor | Lab Research Project")
    html.div(class: "entry-desc", [
      Synthesized high-quality Chinese data using Ultra-FineWeb-zh as source data, following the Nemotron-CC methodology. Continual pre-training of a 100B-token MiniCPM model on 3B synthetic tokens showed significant improvements on MMLU, CMMLU, and C-Eval benchmarks.
    ])
  })

  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "MiniCPM-SALA")
      html.div(class: "entry-time", "2026.01 -- 2026.02")
    })
    html.div(class: "entry-meta", "Contributor | Lab Research Project")
    html.div(class: "entry-desc", "Prepared general knowledge data for the fine-tuning stage of MiniCPM-SALA.")
  })
}

= Experience

#{
  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "Tsinghua University HPC Team")
      html.div(class: "entry-time", "2025.06 -- Present")
    })
    html.div(class: "entry-meta", "Team Member")
  })

  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "Tsinghua University Algorithm Association")
      html.div(class: "entry-time", "2023.09 -- Present")
    })
    html.div(class: "entry-meta", "Core Member, Competition & Platform Division")
  })

  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "NOI 2026 Winter Camp")
      html.div(class: "entry-time", "2026.02")
    })
    html.div(class: "entry-meta", "CCF Student Expert")
  })

  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "Object-Oriented Programming")
      html.div(class: "entry-time", "Spring 2024")
    })
    html.div(class: "entry-meta", "Teaching Assistant")
  })

  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "Zhili-Math&CS Class 31")
      html.div(class: "entry-time", "2023.09 -- Present")
    })
    html.div(class: "entry-meta", "Class Committee (Organizer → Class President → Life Commissioner)")
  })

  html.div(class: "section-entry", {
    html.div(class: "entry-header", {
      html.div(class: "entry-title", "Zhili College Student Union")
      html.div(class: "entry-time", "2023.09 -- 2024.08")
    })
    html.div(class: "entry-meta", "Member, Academic Department")
  })
}

= Competitions & Awards

- *PAC2025* (National Parallel Application Challenge): 1st Place
- *Excellence Scholarship for Sci-Tech Innovation*: 2024, 2025
- *2023 CCPC* (Collegiate Programming Contest, Harbin): 4th Place

= Skills

- *Programming Languages*: C++, Python, Rust
- *Domains*: LLM, HPC, Network
- *Current Interests*: LLM Algorithms, LLM Systems

#{ html.elem("hr")[] }

Blog: #link("/posts/")[Posts] · #link("/tags/")[Tags] · #link("/categories/")[Categories] · #link("/archive/")[Archive]
