# docs/ —— 知识库语料

这个目录**默认不会上传到 GitHub**（见 `.gitignore` 里的 `docs/*`）：
里面通常是公司内部资料或客户文件，不适合公开。

## 想跑起来，把自己的文档放进来

```
docs/
├── 方法论/         ← 一个子目录就是一个可筛选的「文档范围」
│   ├── xxx.md
│   └── yyy.docx
├── 需求文档/
│   └── zzz.pdf
└── 行业报告/
    └── aaa.pptx
```

支持格式：`.txt` `.md` `.html` `.pdf` `.docx` `.pptx`
（`.xlsx` 不支持，把内容贴进 txt/md）

**放好之后**：

```powershell
python build_index.py     # 会打印「共 N 块」
```

然后就能在控制台的「📁 文档范围」里按文件夹筛选检索了。

## 想连语料一起开源？

删掉 `.gitignore` 里这两行即可：

```
docs/*
!docs/README.md
```

⚠️ 上传前请确认：**没有客户名称、报价、内部经营数据、未公开的战略文档**。
