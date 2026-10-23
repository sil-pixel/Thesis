-- Pandoc filter used by build.py: Markdown section files -> body of the Springer Nature (sn-jnl) template.
--   * HTML comments (<!-- src: ... -->, status notes) become LaTeX comments, so provenance stays in the .tex
--     but not in the PDF. With the metadata flag strip_comments they are dropped.
--   * Inline code spans hold the placeholders ([TODO], [VERIFY], [AUTHOR INPUT], [CITATION NEEDED]); they are
--     printed in red so they cannot be missed in the PDF.
--   * A paragraph "**Table N.** caption" directly before a table becomes that table's caption.
--   * Headings inside the back matter (abbreviations, declarations) become \bmhead, as in the template.

local strip_comments = false
local in_backmatter = false

local function comment_text(raw)
  local body = raw:match("^%s*<!%-%-(.-)%-%->%s*$")
  return body
end

local function as_latex_comment(body)
  local lines = {}
  for line in (body .. "\n"):gmatch("(.-)\n") do
    line = line:gsub("^%s+", "")
    if line ~= "" then table.insert(lines, "% " .. line) end
  end
  return table.concat(lines, "\n")
end

function Meta(meta)
  if meta.strip_comments then strip_comments = true end
  if meta.backmatter then in_backmatter = true end
end

function RawBlock(el)
  if el.format ~= "html" then return nil end
  local body = comment_text(el.text)
  if not body then return nil end
  if strip_comments then return {} end
  return pandoc.RawBlock("latex", as_latex_comment(body))
end

function RawInline(el)
  if el.format ~= "html" then return nil end
  local body = comment_text(el.text)
  if not body then return nil end
  if strip_comments then return {} end
  return pandoc.RawInline("latex", as_latex_comment(body) .. "\n")
end

function Code(el)
  return { pandoc.RawInline("latex", "\\textcolor{red}{"), pandoc.Str(el.text), pandoc.RawInline("latex", "}") }
end

function Header(el)
  if in_backmatter then
    local text = pandoc.utils.stringify(el.content)
    if el.level == 1 and text == "Declarations" then
      return pandoc.RawBlock("latex", "\\section*{Declarations}")
    elseif el.level == 1 and text == "Figure legends" then
      return pandoc.RawBlock("latex", "\\section*{Figure legends}")
    else
      return pandoc.RawBlock("latex", "\\bmhead{" .. text .. "}")
    end
  end
  return nil
end

local function table_caption(block)
  if block.t ~= "Para" or #block.content == 0 then return nil end
  local first = block.content[1]
  if first.t ~= "Strong" then return nil end
  local label = pandoc.utils.stringify(first.content)
  if not label:match("^Table %d+%.$") then return nil end
  local rest = {}
  for i = 2, #block.content do table.insert(rest, block.content[i]) end
  while #rest > 0 and rest[1].t == "Space" do table.remove(rest, 1) end
  return rest
end

function Blocks(blocks)
  local out = {}
  local i = 1
  while i <= #blocks do
    local block = blocks[i]
    local caption = table_caption(block)
    local nxt = blocks[i + 1]
    -- allow one comment block between the caption paragraph and the table
    local skip = 1
    if caption and nxt and nxt.t == "RawBlock" and blocks[i + 2] and blocks[i + 2].t == "Table" then
      table.insert(out, nxt)
      nxt = blocks[i + 2]
      skip = 2
    end
    if caption and nxt and nxt.t == "Table" then
      nxt.caption = { long = { pandoc.Plain(caption) } }
      table.insert(out, nxt)
      i = i + skip + 1
    else
      table.insert(out, block)
      i = i + 1
    end
  end
  return out
end

return {
  { Meta = Meta },
  { RawBlock = RawBlock, RawInline = RawInline, Code = Code, Header = Header },
  { Blocks = Blocks },
}
