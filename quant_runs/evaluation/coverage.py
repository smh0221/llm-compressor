"""量化覆盖预览 —— 打印各 Linear 是否被量化及命中的忽略规则。

给定已加载模型 + recipe（GPTQModifier list），复用 compressed_tensors 的匹配器算出
[Q]/[skip] 集，按模块层级打印一棵树 + 统计摘要，随后由调用方继续正常量化。
结构+标签完全同构的相邻兄弟子树折叠为「代表 ×N」，避免逐专家/逐层刷屏。
可选写出自包含 HTML（<details> 折叠/展开），路径由 out_path 指定。
"""

import html

from compressed_tensors.utils import match_named_modules, match_targets

# 叶子行标签对齐列，够宽以容纳逐专家长名（如 layers.N.mlp.experts.M.gate_proj）
LEAF_COL = 46

# 按层类型分布的固定展示顺序
LAYER_TYPES = ("attn", "mlp", "expert", "other")


def preview_coverage(model, recipe, out_path=None):
    # 主入口，只处理 recipe[0]（当前脚本单 modifier），recipe 为空则跳过
    if not recipe:
        return

    modifier = recipe[0]
    targets = modifier.targets
    ignore = modifier.ignore
    scheme = modifier.scheme

    cov = collect_coverage(model, targets, ignore)
    model_name = model.__class__.__name__

    # 打印 header，含 model 类名 / scheme / targets / ignore 规则列表
    print("=== Quant Coverage Preview ===")
    print(f"Model: {model_name}")
    print(f"Scheme: {scheme}   Targets: {list(targets)}")
    print(f"Ignore: {' | '.join(ignore) if ignore else '(none)'}")

    print("--- Module tree ---")
    render_tree(cov)

    print_summary(cov, ignore)

    # 可选写出可交互 HTML，同构组默认收起，点开逐个展开
    if out_path:
        write_html(out_path, cov, ignore, model_name, scheme, targets)
        print(f"Coverage HTML written to: {out_path}")


def collect_coverage(model, targets, ignore):
    # 分母（全部命中 targets 的 Linear），ignore 置空以复用匹配器自身的类名判定
    total = dict(match_named_modules(model, targets, ignore=[]))
    # [Q] 集，与量化判定同源
    quantized = dict(match_named_modules(model, targets, ignore=ignore))
    # [skip] = 分母 − [Q]
    ignored = {name: total[name] for name in total if name not in quantized}

    # 每个被 skip 的模块命中的最具体规则，match_targets 返回按具体度排序，取第一条
    skip_rule = {}
    for name, module in ignored.items():
        matched = match_targets(name, module, ignore)
        skip_rule[name] = matched[0] if matched else "?"

    # 按 skip_rule 计每条规则命中数，和 = len(ignored)
    rule_hits = {}
    for rule in skip_rule.values():
        rule_hits[rule] = rule_hits.get(rule, 0) + 1

    return {
        "total": total,
        "quantized": quantized,
        "ignored": ignored,
        "skip_rule": skip_rule,
        "rule_hits": rule_hits,
    }


def classify_layer_type(name):
    # 按名字子串分桶，用于按类型统计 Q/skip 分布
    # experts 优先判定，避免逐专家 mlp.experts.N.* 落入 mlp 桶
    if "experts" in name:
        return "expert"
    if "self_attn" in name or "attn" in name:
        return "attn"
    if (
        "mlp" in name
        or "shared_expert" in name
        or "gate" in name
        or "up_proj" in name
        or "down_proj" in name
    ):
        return "mlp"
    return "other"


def build_tree(names):
    # 把点分名建成嵌套 dict，叶子用 None 占位（真实叶子为 Linear 全名）
    root = {}
    for name in names:
        node = root
        parts = name.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(parts[-1], None)
    return root


def subtree_signature(node, prefix, quantized, skip_rule):
    # 递归算子树「形状+标签」指纹，用于判定兄弟子树是否同构可折叠
    # 叶子指纹为其量化标签（[Q] 或 skip 规则），内部节点为 (子名, 子指纹) 有序元组
    if node is None:
        if prefix in quantized:
            return "Q"
        return f"skip:{skip_rule.get(prefix, '?')}"
    items = []
    for key, child in node.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        items.append((key, subtree_signature(child, child_prefix, quantized, skip_rule)))
    return tuple(items)


def group_children(node, prefix, quantized, skip_rule):
    # 把子节点按「形状+标签」指纹分连续组，同指纹的相邻兄弟折叠
    # 返回 [(sig, [(key, child, child_prefix), ...]), ...]，保序（named_modules 顺序）
    groups = []
    for key, child in node.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        sig = subtree_signature(child, child_prefix, quantized, skip_rule)
        if groups and groups[-1][0] == sig:
            groups[-1][1].append((key, child, child_prefix))
        else:
            groups.append((sig, [(key, child, child_prefix)]))
    return groups


def group_display_name(members):
    # 折叠组用首末成员名概括，如 0..255；单成员保持原名
    if len(members) > 1:
        return f"{members[0][0]}..{members[-1][0]} ×{len(members)}"
    return members[0][0]


def leaf_label(full_name, quantized, skip_rule):
    if full_name in quantized:
        return "[Q]"
    return f"[skip: {skip_rule.get(full_name, '?')}]"


def render_tree(cov):
    # 渲染整棵树到终端，同构兄弟子树折叠为「代表 ×N」，叶子标 [Q] 或 [skip: <rule>]
    quantized = cov["quantized"]
    skip_rule = cov["skip_rule"]
    tree = build_tree(list(cov["total"].keys()))

    def walk(node, prefix, indent):
        groups = group_children(node, prefix, quantized, skip_rule)
        for gi, (_sig, members) in enumerate(groups):
            last_group = gi == len(groups) - 1
            branch = "└─ " if last_group else "├─ "
            rep_key, rep_child, rep_prefix = members[0]
            disp = group_display_name(members)
            if rep_child is None:
                # 叶子（组），标注 [Q]/[skip] 并按显示宽度右侧对齐
                line = f"{indent}{branch}{disp}"
                pad = max(1, LEAF_COL - len(line))
                print(f"{line}{' ' * pad}{leaf_label(rep_prefix, quantized, skip_rule)}")
            else:
                print(f"{indent}{branch}{disp}")
                next_indent = indent + ("   " if last_group else "│  ")
                walk(rep_child, rep_prefix, next_indent)

    walk(tree, "", "")


def print_summary(cov, ignore):
    # 打印摘要，总数 + 每条 ignore 规则命中表 + 按层类型的 Q/skip 分布表
    total = cov["total"]
    quantized = cov["quantized"]
    ignored = cov["ignored"]
    rule_hits = cov["rule_hits"]

    print("--- Summary ---")
    print(f"Total Linear:   {len(total)}")
    print(f"Quantized [Q]:  {len(quantized)}")
    print(f"Ignored [skip]: {len(ignored)}")

    # 每条 ignore 规则命中数，原样打印规则；未命中任何模块的规则也列出为 0
    print("Per-ignore-rule hits:")
    width = max((len(r) for r in ignore), default=0)
    for rule in ignore:
        print(f"  {rule:<{width}}  {rule_hits.get(rule, 0)}")
    # 兜底桶，正常不应出现，仅当某 skip 模块未匹配到任何规则时才有
    if "?" in rule_hits:
        print(f"  {'?':<{width}}  {rule_hits['?']}")

    # 按层类型统计 Q/skip 分布，分子来自各集合按 classify_layer_type 归类
    q_by_type, s_by_type = compute_layer_type_dist(quantized, ignored)

    print("By layer type (Q / skip):")
    for t in LAYER_TYPES:
        print(f"  {t:<8} {q_by_type[t]} / {s_by_type[t]}")


def compute_layer_type_dist(quantized, ignored):
    # 按层类型统计 Q/skip 分布，供终端与 HTML 共用
    q_by_type = {t: 0 for t in LAYER_TYPES}
    s_by_type = {t: 0 for t in LAYER_TYPES}
    for name in quantized:
        q_by_type[classify_layer_type(name)] += 1
    for name in ignored:
        s_by_type[classify_layer_type(name)] += 1
    return q_by_type, s_by_type


def _tree_to_html(node, prefix, quantized, skip_rule):
    # 递归把折叠树渲染为嵌套 <details>，同构组默认收起（无 open 属性）
    # 组内多成员时 <summary> 标 ×N，展开后逐个成员各自成一棵子树
    parts = []
    for _sig, members in group_children(node, prefix, quantized, skip_rule):
        rep_key, rep_child, rep_prefix = members[0]
        disp = html.escape(group_display_name(members))
        collapsed = len(members) > 1
        if rep_child is None:
            # 叶子组，量化标签决定 CSS 类，[Q] 绿 / [skip] 灰
            label = leaf_label(rep_prefix, quantized, skip_rule)
            cls = "q" if rep_prefix in quantized else "skip"
            leaf_html = (
                f'<div class="leaf"><span class="name">{disp}</span>'
                f'<span class="tag {cls}">{html.escape(label)}</span></div>'
            )
            if collapsed:
                # 折叠的叶子组，点开逐个列出成员（含真实序号名 + 各自标签）
                inner = []
                for key, _c, cp in members:
                    m_label = leaf_label(cp, quantized, skip_rule)
                    m_cls = "q" if cp in quantized else "skip"
                    inner.append(
                        f'<div class="leaf"><span class="name">'
                        f'{html.escape(key)}</span>'
                        f'<span class="tag {m_cls}">'
                        f'{html.escape(m_label)}</span></div>'
                    )
                parts.append(
                    f"<details><summary>{disp} "
                    f'<span class="tag {cls}">{html.escape(label)}</span>'
                    f'</summary><div class="children">{"".join(inner)}'
                    f"</div></details>"
                )
            else:
                parts.append(leaf_html)
        else:
            # 内部节点组，代表子树递归渲染；折叠组标 ×N 并只展开代表结构
            child_html = _tree_to_html(rep_child, rep_prefix, quantized, skip_rule)
            open_attr = "" if collapsed else " open"
            parts.append(
                f"<details{open_attr}><summary>{disp}</summary>"
                f'<div class="children">{child_html}</div></details>'
            )
    return "".join(parts)


# HTML 样式，纯 CSS 无 JS，<details> 原生折叠；[Q] 绿 / [skip] 灰
_HTML_STYLE = """
body{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
background:#1e1e1e;color:#d4d4d4;margin:0;padding:16px;font-size:13px}
h1{font-size:16px;margin:0 0 8px}
.meta{color:#9cdcfe;margin-bottom:4px;white-space:pre-wrap;word-break:break-all}
.controls{margin:10px 0}
.controls button{background:#333;color:#d4d4d4;border:1px solid #555;
border-radius:4px;padding:3px 10px;cursor:pointer;margin-right:6px;font-size:12px}
.controls button:hover{background:#444}
details{margin-left:14px;border-left:1px dotted #444;padding-left:8px}
summary{cursor:pointer;list-style:none;padding:1px 0}
summary::-webkit-details-marker{display:none}
summary::before{content:"\\25B8 ";color:#888}
details[open]>summary::before{content:"\\25BE ";color:#888}
.children{}
.leaf{margin-left:14px;padding:1px 0;display:flex;justify-content:space-between;
max-width:640px}
.name{color:#d4d4d4}
.tag{font-weight:bold;margin-left:12px}
.tag.q{color:#4ec9b0}
.tag.skip{color:#c586c0}
table{border-collapse:collapse;margin-top:8px}
td,th{border:1px solid #444;padding:2px 10px;text-align:left}
th{color:#9cdcfe}
.sec{margin-top:18px;font-size:14px;color:#dcdcaa}
"""


def _summary_html(cov, ignore):
    # 摘要区，总数 + 每规则命中表 + 层类型分布表
    total, quantized, ignored = cov["total"], cov["quantized"], cov["ignored"]
    rule_hits = cov["rule_hits"]
    rows = [
        f"<div>Total Linear: <b>{len(total)}</b> &nbsp; "
        f'Quantized [Q]: <b class="tag q">{len(quantized)}</b> &nbsp; '
        f'Ignored [skip]: <b class="tag skip">{len(ignored)}</b></div>'
    ]
    rows.append('<div class="sec">Per-ignore-rule hits</div>')
    rows.append("<table><tr><th>rule</th><th>hits</th></tr>")
    for rule in ignore:
        rows.append(
            f"<tr><td>{html.escape(rule)}</td>"
            f"<td>{rule_hits.get(rule, 0)}</td></tr>"
        )
    if "?" in rule_hits:
        rows.append(f"<tr><td>?</td><td>{rule_hits['?']}</td></tr>")
    rows.append("</table>")

    q_by_type, s_by_type = compute_layer_type_dist(quantized, ignored)
    rows.append('<div class="sec">By layer type</div>')
    rows.append("<table><tr><th>type</th><th>Q</th><th>skip</th></tr>")
    for t in LAYER_TYPES:
        rows.append(
            f"<tr><td>{t}</td><td>{q_by_type[t]}</td>"
            f"<td>{s_by_type[t]}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def write_html(out_path, cov, ignore, model_name, scheme, targets):
    # 写出自包含 HTML，嵌套 <details> 折叠/展开，同构组默认收起
    quantized = cov["quantized"]
    skip_rule = cov["skip_rule"]
    tree = build_tree(list(cov["total"].keys()))
    tree_html = _tree_to_html(tree, "", quantized, skip_rule)

    ignore_line = " | ".join(ignore) if ignore else "(none)"
    # 全部展开/收起按钮，纯内联 JS 遍历 <details> 切换 open
    script = (
        "<script>function setAll(o){"
        "document.querySelectorAll('details').forEach(function(d){d.open=o});}"
        "</script>"
    )
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Quant Coverage — {html.escape(model_name)}</title>"
        f"<style>{_HTML_STYLE}</style></head><body>"
        "<h1>Quant Coverage Preview</h1>"
        f'<div class="meta">Model: {html.escape(model_name)}</div>'
        f'<div class="meta">Scheme: {html.escape(str(scheme))}   '
        f"Targets: {html.escape(str(list(targets)))}</div>"
        f'<div class="meta">Ignore: {html.escape(ignore_line)}</div>'
        '<div class="controls">'
        "<button onclick=\"setAll(true)\">展开全部</button>"
        "<button onclick=\"setAll(false)\">收起全部</button></div>"
        f'<div class="sec">Module tree</div><div id="tree">{tree_html}</div>'
        f"{_summary_html(cov, ignore)}"
        f"{script}</body></html>"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)