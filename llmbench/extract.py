"""Document text extraction — turns real-world files (PDF, Word, PowerPoint,
Excel, RTF, HTML, ODT, plain text) into plain text that can be injected into a
prompt as reference material.

Design: zero third-party dependencies. Where a high-quality system converter is
available (`pdftotext`, `textutil`) it is used; otherwise a stdlib-only fallback
runs so the harness keeps working on any machine. Each extraction reports the
method used and any quality warnings, which the report surfaces so results stay
honest.

Known limits: the built-in PDF parser handles text-based PDFs (FlateDecode
content streams, standard encodings). Scanned/image PDFs and exotic CID-font
encodings need `pdftotext` (poppler) or OCR — the extractor flags these cases.
"""
from __future__ import annotations

import html as html_mod
import os
import re
import shutil
import subprocess
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import List, Optional
from xml.etree import ElementTree as ET

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".json", ".jsonl", ".csv", ".tsv",
    ".py", ".js", ".ts", ".tsx", ".java", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".sh", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".sql", ".log", ".tex",
}
DOC_EXTS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".rtf", ".doc", ".html", ".htm",
}


@dataclass
class Extraction:
    text: str
    method: str
    warnings: List[str] = field(default_factory=list)


def supported(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in TEXT_EXTS or ext in DOC_EXTS


def extract_document(path: str) -> Extraction:
    """Dispatch on extension. Returns extracted text + method + warnings."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return _pdf(path)
        if ext == ".docx":
            return _docx(path)
        if ext == ".pptx":
            return _pptx(path)
        if ext == ".xlsx":
            return _xlsx(path)
        if ext == ".odt":
            return _odt(path)
        if ext == ".rtf":
            return _rtf(path)
        if ext == ".doc":
            return _legacy_doc(path)
        if ext in (".html", ".htm"):
            return _html(path)
        return _plain(path)
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001 - never let one bad file kill a run
        return Extraction("", f"{ext or 'file'} (failed)",
                          [f"extraction error: {type(e).__name__}: {e}"])


# --------------------------------------------------------------------------
# Plain text / HTML
# --------------------------------------------------------------------------
def _plain(path: str) -> Extraction:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return Extraction(f.read(), "text")


def _html(path: str) -> Extraction:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)>", "\n", raw)
    text = re.sub(r"<[^>]+>", "", raw)
    text = html_mod.unescape(text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return Extraction(text.strip(), "html (tag-strip)")


# --------------------------------------------------------------------------
# OOXML (docx/pptx/xlsx) + ODT — all zipped XML, handled with stdlib
# --------------------------------------------------------------------------
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_SS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _docx(path: str) -> Extraction:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras: List[str] = []
    for p in root.iter(_W + "p"):
        buf: List[str] = []
        for node in p.iter():
            if node.tag == _W + "t" and node.text:
                buf.append(node.text)
            elif node.tag == _W + "tab":
                buf.append("\t")
            elif node.tag in (_W + "br", _W + "cr"):
                buf.append("\n")
        paras.append("".join(buf))
    return Extraction("\n".join(paras).strip(), "docx (stdlib)")


def _pptx(path: str) -> Extraction:
    texts: List[str] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist()
                 if re.match(r"ppt/slides/slide\d+\.xml$", n)]
        names.sort(key=lambda n: int(re.search(r"(\d+)", n).group(1)))
        for n in names:
            root = ET.fromstring(z.read(n))
            slide = [t.text for t in root.iter(_A + "t") if t.text]
            if slide:
                texts.append(" ".join(slide))
    return Extraction("\n\n".join(texts).strip(), "pptx (stdlib)")


def _xlsx(path: str) -> Extraction:
    with zipfile.ZipFile(path) as z:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.iter(_SS + "si"):
                shared.append("".join(t.text or "" for t in si.iter(_SS + "t")))
        rows: List[str] = []
        sheets = sorted(n for n in z.namelist()
                        if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
        for sh in sheets:
            root = ET.fromstring(z.read(sh))
            for row in root.iter(_SS + "row"):
                cells: List[str] = []
                for c in row.iter(_SS + "c"):
                    v = c.find(_SS + "v")
                    if v is None or v.text is None:
                        continue
                    if c.get("t") == "s":  # shared-string index
                        idx = int(v.text)
                        cells.append(shared[idx] if idx < len(shared) else "")
                    else:
                        cells.append(v.text)
                if cells:
                    rows.append("\t".join(cells))
    warn = [] if rows else ["no cell values found"]
    return Extraction("\n".join(rows).strip(), "xlsx (stdlib)", warn)


def _odt(path: str) -> Extraction:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    # ODT text lives in <text:p>/<text:span>; grab all element text in order.
    parts = [t for t in root.itertext()]
    text = "\n".join(s for s in (p.strip() for p in parts) if s)
    return Extraction(text.strip(), "odt (stdlib)")


# --------------------------------------------------------------------------
# RTF / legacy DOC — prefer macOS `textutil`, fall back to a stripper
# --------------------------------------------------------------------------
def _textutil(path: str) -> Optional[str]:
    if shutil.which("textutil") is None:
        return None
    try:
        out = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", path],
            capture_output=True, text=True, timeout=30, check=False)
        if out.returncode == 0:
            return out.stdout
    except Exception:
        pass
    return None


def _rtf(path: str) -> Extraction:
    via = _textutil(path)
    if via is not None:
        return Extraction(via.strip(), "rtf (textutil)")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    # Minimal RTF strip: drop groups like {\*\...}, control words, braces.
    raw = re.sub(r"\\'([0-9a-fA-F]{2})",
                 lambda m: bytes([int(m.group(1), 16)]).decode("latin-1"), raw)
    raw = re.sub(r"\\par[d]?", "\n", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    raw = raw.replace("{", "").replace("}", "")
    return Extraction(raw.strip(), "rtf (stdlib strip)",
                      ["basic RTF stripper; install macOS textutil for fidelity"])


def _legacy_doc(path: str) -> Extraction:
    via = _textutil(path)
    if via is not None:
        return Extraction(via.strip(), "doc (textutil)")
    return Extraction("", "doc (unsupported)",
                      ["legacy .doc needs macOS textutil or conversion to .docx/.txt"])


# --------------------------------------------------------------------------
# PDF — external pdftotext preferred, stdlib fallback otherwise
# --------------------------------------------------------------------------
def _pdf(path: str) -> Extraction:
    if shutil.which("pdftotext") is not None:
        try:
            out = subprocess.run(["pdftotext", "-q", "-enc", "UTF-8", path, "-"],
                                 capture_output=True, text=True, timeout=60, check=False)
            if out.returncode == 0 and out.stdout.strip():
                return Extraction(out.stdout.strip(), "pdf (pdftotext)")
        except Exception:
            pass
    return _pdf_builtin(path)


def _pdf_builtin(path: str) -> Extraction:
    with open(path, "rb") as f:
        data = f.read()
    warnings: List[str] = []
    pieces: List[str] = []
    for m in re.finditer(rb"stream\r?\n", data):
        start = m.end()
        end = data.find(b"endstream", start)
        if end == -1:
            continue
        raw = data[start:end].rstrip(b"\r\n")
        decoded: Optional[bytes] = None
        try:
            decoded = zlib.decompress(raw)
        except Exception:
            if b"BT" in raw or b"Tj" in raw or b"TJ" in raw:
                decoded = raw  # uncompressed content stream
        if decoded is None:
            continue
        piece = _pdf_content_text(decoded)
        if piece.strip():
            pieces.append(piece)
    text = "\n".join(pieces)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        warnings.append("no extractable text: likely a scanned/image PDF or "
                        "custom font encoding; install poppler `pdftotext` or OCR it")
    else:
        printable = sum(c.isprintable() or c in "\n\t" for c in text)
        if printable / max(1, len(text)) < 0.7:
            warnings.append("low-confidence extraction (unusual font encoding); "
                            "verify against `pdftotext` output")
    return Extraction(text, "pdf (builtin)", warnings)


def _pdf_content_text(s: bytes) -> str:
    """Pull visible text out of a PDF content stream. Collects string operands of
    text-showing operators inside BT/ET blocks and inserts newlines at line-
    positioning operators."""
    out: List[str] = []
    i, n = 0, len(s)
    in_text = False
    while i < n:
        ch = s[i]
        if ch == 0x28:  # '(' literal string
            depth, i = 1, i + 1
            buf = bytearray()
            while i < n and depth > 0:
                cc = s[i]
                if cc == 0x5C and i + 1 < n:  # backslash escape
                    buf.append(cc)
                    buf.append(s[i + 1])
                    i += 2
                    continue
                if cc == 0x28:
                    depth += 1
                elif cc == 0x29:
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                buf.append(cc)
                i += 1
            if in_text:
                out.append(_decode_pdf_string(bytes(buf)))
            continue
        if ch == 0x3C and i + 1 < n and s[i + 1] != 0x3C:  # '<' hex string
            j = s.find(b">", i + 1)
            if j != -1:
                hexs = re.sub(rb"\s+", b"", s[i + 1:j])
                if len(hexs) % 2:
                    hexs += b"0"
                try:
                    if in_text:
                        out.append(bytes.fromhex(hexs.decode("ascii")).decode("latin-1"))
                except Exception:
                    pass
                i = j + 1
                continue
        two = s[i:i + 2]
        if two == b"BT":
            in_text = True
            i += 2
            continue
        if two == b"ET":
            in_text = False
            out.append("\n")
            i += 2
            continue
        if two in (b"Td", b"TD", b"T*"):
            out.append("\n")
            i += 2
            continue
        i += 1
    return "".join(out)


_ESCAPES = {0x6E: 0x0A, 0x72: 0x0D, 0x74: 0x09, 0x62: 0x08, 0x66: 0x0C,
            0x28: 0x28, 0x29: 0x29, 0x5C: 0x5C}


def _decode_pdf_string(b: bytes) -> str:
    out = bytearray()
    i, n = 0, len(b)
    while i < n:
        c = b[i]
        if c == 0x5C and i + 1 < n:  # backslash
            nxt = b[i + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
            if 0x30 <= nxt <= 0x37:  # octal escape, up to 3 digits
                mo = re.match(rb"[0-7]{1,3}", b[i + 1:i + 4])
                out.append(int(mo.group(0), 8) & 0xFF)
                i += 1 + len(mo.group(0))
                continue
            if nxt in (0x0A, 0x0D):  # line continuation
                i += 2
                continue
            out.append(nxt)
            i += 2
            continue
        out.append(c)
        i += 1
    return out.decode("latin-1")
