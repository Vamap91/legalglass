"""
LegalGlass AI
Inteligência Artificial para Gestão Contratual e Análise Jurídica

Plataforma com dois módulos:
  - Módulo 1: Comparador de Contratos
  - Módulo 2: Chat Jurídico Inteligente (RAG sobre base de contratos)

Stack: Streamlit + OpenAI + FAISS
"""

import io
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from openai import OpenAI

# Extração de documentos
from pypdf import PdfReader
import docx as docxlib

# FAISS
import faiss

# Geração de PDF executivo
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Ajuste o nome do modelo conforme disponibilidade na sua conta OpenAI.
# A especificação cita "GPT-5"; o flagship atual da API é a família GPT-5.5.
CHAT_MODEL = "gpt-5.5"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536          # dimensão do text-embedding-3-small
CHUNK_SIZE = 1200             # caracteres por bloco
CHUNK_OVERLAP = 200           # sobreposição entre blocos
TOP_K = 6                     # trechos recuperados por pergunta
MAX_CONTRACTS = 30

st.set_page_config(
    page_title="LegalGlass AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cliente OpenAI
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_client() -> Optional[OpenAI]:
    """Cria o cliente OpenAI a partir da chave em Streamlit Secrets."""
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Extração de texto
# ---------------------------------------------------------------------------

def extract_pdf_pages(file_bytes: bytes) -> List[str]:
    """Retorna uma lista com o texto de cada página."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def extract_docx_pages(file_bytes: bytes) -> List[str]:
    """DOCX não tem paginação confiável; tratamos o documento como 'página 1'."""
    document = docxlib.Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [text]


def extract_pages(uploaded_file) -> List[str]:
    """Extrai páginas de um arquivo enviado (PDF ou DOCX)."""
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_pdf_pages(data)
    if name.endswith(".docx"):
        return extract_docx_pages(data)
    raise ValueError(f"Formato não suportado: {uploaded_file.name}")


def extract_full_text(uploaded_file) -> str:
    return "\n".join(extract_pages(uploaded_file))


# ---------------------------------------------------------------------------
# Chunking + Embeddings + FAISS (Módulo 2)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    contract: str
    page: int
    text: str


@dataclass
class VectorStore:
    index: Any = None
    chunks: List[Chunk] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.index is None or not self.chunks


def chunk_pages(contract_name: str, pages: List[str]) -> List[Chunk]:
    """Divide o texto em blocos com sobreposição, preservando a página de origem."""
    chunks: List[Chunk] = []
    for page_num, page_text in enumerate(pages, start=1):
        text = (page_text or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(contract=contract_name, page=page_num, text=piece))
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP
    return chunks


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Gera embeddings em lote e retorna matriz normalizada (para cosseno via IP)."""
    vectors: List[List[float]] = []
    batch = 96
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[i:i + batch],
        )
        vectors.extend([d.embedding for d in resp.data])
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_vector_store(client: OpenAI, chunks: List[Chunk]) -> VectorStore:
    """Constrói o índice FAISS a partir dos blocos."""
    if not chunks:
        return VectorStore()
    embeddings = embed_texts(client, [c.text for c in chunks])
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return VectorStore(index=index, chunks=chunks)


def search(client: OpenAI, store: VectorStore, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Recupera os trechos mais relevantes para a pergunta."""
    q = embed_texts(client, [query])
    scores, idxs = store.index.search(q, min(top_k, len(store.chunks)))
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        c = store.chunks[idx]
        results.append({
            "contract": c.contract,
            "page": c.page,
            "text": c.text,
            "score": float(score),
        })
    return results


# ---------------------------------------------------------------------------
# Chamadas de IA
# ---------------------------------------------------------------------------

def chat_completion(client: OpenAI, system: str, user: str,
                    json_mode: bool = False) -> str:
    kwargs: Dict[str, Any] = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Módulo 1 — Comparador de Contratos
# ---------------------------------------------------------------------------

COMPARE_SYSTEM = """Você é um analista jurídico sênior especializado em análise contratual.
Compare a versão ANTIGA e a versão ATUAL de um mesmo contrato e identifique todas as
alterações relevantes. Considere especialmente: cláusulas adicionadas, removidas e
alteradas; mudanças financeiras, de responsabilidade, de prazo, de multas, de SLA e de LGPD;
e riscos jurídicos decorrentes.

Responda EXCLUSIVAMENTE em JSON válido com a estrutura:
{
  "resumo_executivo": "string com uma frase objetiva, ex.: 'Identificadas 14 alterações relevantes entre as versões.'",
  "total_alteracoes": int,
  "classificacao_risco": "Baixo" | "Médio" | "Alto",
  "justificativa_risco": "string curta",
  "alteracoes": [
    {
      "tipo": "Adicionada" | "Removida" | "Alterada",
      "clausula": "string (ex.: 'Cláusula 8')",
      "categoria": "Financeira|Responsabilidade|Prazo|Multa|SLA|LGPD|Garantia|Outro",
      "descricao": "string objetiva da alteração",
      "risco": "Baixo" | "Médio" | "Alto"
    }
  ]
}
Não inclua texto fora do JSON."""


def compare_contracts(client: OpenAI, old_text: str, new_text: str) -> Dict[str, Any]:
    # Limita o tamanho enviado para evitar estouro de contexto em contratos muito longos.
    limit = 60000
    user = (
        "CONTRATO ANTIGO:\n"
        f"{old_text[:limit]}\n\n"
        "CONTRATO ATUAL:\n"
        f"{new_text[:limit]}\n\n"
        "Gere a análise comparativa no formato JSON especificado."
    )
    raw = chat_completion(client, COMPARE_SYSTEM, user, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)


def build_comparison_df(result: Dict[str, Any]) -> pd.DataFrame:
    rows = result.get("alteracoes", [])
    if not rows:
        return pd.DataFrame(columns=["Tipo", "Cláusula", "Categoria", "Alteração", "Risco"])
    return pd.DataFrame([{
        "Tipo": r.get("tipo", ""),
        "Cláusula": r.get("clausula", ""),
        "Categoria": r.get("categoria", ""),
        "Alteração": r.get("descricao", ""),
        "Risco": r.get("risco", ""),
    } for r in rows])


def build_excel(df: pd.DataFrame, result: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary = pd.DataFrame({
            "Campo": ["Resumo Executivo", "Total de Alterações", "Classificação de Risco", "Justificativa"],
            "Valor": [
                result.get("resumo_executivo", ""),
                result.get("total_alteracoes", len(df)),
                result.get("classificacao_risco", ""),
                result.get("justificativa_risco", ""),
            ],
        })
        summary.to_excel(writer, index=False, sheet_name="Resumo")
        df.to_excel(writer, index=False, sheet_name="Comparativo")
    return buffer.getvalue()


def build_pdf(df: pd.DataFrame, result: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=18)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13)
    normal = styles["BodyText"]

    risk = result.get("classificacao_risco", "—")
    risk_color = {"Baixo": colors.green, "Médio": colors.orange, "Alto": colors.red}.get(risk, colors.black)

    story = [
        Paragraph("LegalGlass AI — Resumo Executivo", title_style),
        Spacer(1, 12),
        Paragraph(result.get("resumo_executivo", ""), normal),
        Spacer(1, 8),
        Paragraph(f"<b>Total de alterações:</b> {result.get('total_alteracoes', len(df))}", normal),
        Paragraph(
            f'<b>Classificação de risco:</b> <font color="{risk_color.hexval()}">{risk}</font>',
            normal,
        ),
        Paragraph(f"<b>Justificativa:</b> {result.get('justificativa_risco', '')}", normal),
        Spacer(1, 16),
        Paragraph("Tabela Comparativa", h2),
        Spacer(1, 8),
    ]

    data = [["Tipo", "Cláusula", "Categoria", "Alteração", "Risco"]]
    for _, row in df.iterrows():
        data.append([
            Paragraph(str(row["Tipo"]), normal),
            Paragraph(str(row["Cláusula"]), normal),
            Paragraph(str(row["Categoria"]), normal),
            Paragraph(str(row["Alteração"]), normal),
            Paragraph(str(row["Risco"]), normal),
        ])

    table = Table(data, colWidths=[2.2 * cm, 2.3 * cm, 3 * cm, 6 * cm, 1.8 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2a44")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f8")]),
    ]))
    story.append(table)

    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Módulo 2 — Chat Jurídico
# ---------------------------------------------------------------------------

CHAT_SYSTEM = """Você é um assistente jurídico especializado em análise de contratos.
Responda à pergunta do usuário usando APENAS os trechos de contratos fornecidos como contexto.
Se a informação não estiver no contexto, diga claramente que não foi encontrada.

Responda em JSON válido:
{
  "resposta": "resposta objetiva à pergunta",
  "contrato": "nome do contrato mais relevante ou 'N/A'",
  "pagina": "número da página ou 'N/A'",
  "trecho": "trecho exato que fundamenta a resposta",
  "confianca": "Alta" | "Média" | "Baixa"
}
Não inclua texto fora do JSON."""


def answer_question(client: OpenAI, store: VectorStore, question: str) -> Dict[str, Any]:
    hits = search(client, store, question)
    context = "\n\n".join(
        f"[Contrato: {h['contract']} | Página: {h['page']}]\n{h['text']}"
        for h in hits
    )
    user = f"CONTEXTO:\n{context}\n\nPERGUNTA: {question}"
    raw = chat_completion(client, CHAT_SYSTEM, user, json_mode=True)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"resposta": raw, "contrato": "N/A", "pagina": "N/A",
                  "trecho": "", "confianca": "Baixa"}
    parsed["_fontes"] = hits
    return parsed


# ---------------------------------------------------------------------------
# Estado de sessão
# ---------------------------------------------------------------------------

def init_state():
    st.session_state.setdefault("vector_store", VectorStore())
    st.session_state.setdefault("indexed_contracts", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("comparison_result", None)


# ---------------------------------------------------------------------------
# UI — Módulo 1
# ---------------------------------------------------------------------------

def render_comparator(client: OpenAI):
    st.header("Módulo 1 — Comparador de Contratos")
    st.caption("Compare duas versões de um mesmo contrato e identifique as alterações relevantes.")

    col1, col2 = st.columns(2)
    with col1:
        old_file = st.file_uploader("Contrato Antigo", type=["pdf", "docx"], key="old")
    with col2:
        new_file = st.file_uploader("Contrato Atual", type=["pdf", "docx"], key="new")

    if st.button("Comparar contratos", type="primary", disabled=not (old_file and new_file)):
        with st.spinner("Analisando alterações com IA..."):
            old_text = extract_full_text(old_file)
            new_text = extract_full_text(new_file)
            st.session_state.comparison_result = compare_contracts(client, old_text, new_text)

    result = st.session_state.comparison_result
    if not result:
        return

    st.subheader("Resumo Executivo")
    st.info(result.get("resumo_executivo", ""))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de alterações", result.get("total_alteracoes", "—"))
    risk = result.get("classificacao_risco", "—")
    c2.metric("Classificação de risco", risk)
    risk_emoji = {"Baixo": "🟢", "Médio": "🟡", "Alto": "🔴"}.get(risk, "⚪")
    c3.metric("Nível", risk_emoji)
    if result.get("justificativa_risco"):
        st.caption(result["justificativa_risco"])

    st.subheader("Tabela Comparativa")
    df = build_comparison_df(result)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    d1.download_button(
        "📄 PDF Executivo",
        data=build_pdf(df, result),
        file_name="legalglass_resumo_executivo.pdf",
        mime="application/pdf",
    )
    d2.download_button(
        "📊 Excel Comparativo",
        data=build_excel(df, result),
        file_name="legalglass_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------------------------------------------------------------
# UI — Módulo 2
# ---------------------------------------------------------------------------

def render_chat(client: OpenAI):
    st.header("Módulo 2 — Chat Jurídico Inteligente")
    st.caption(f"Faça consultas em linguagem natural sobre sua base de contratos (até {MAX_CONTRACTS}).")

    files = st.file_uploader(
        "Base de contratos (upload múltiplo)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="base",
    )

    if files and len(files) > MAX_CONTRACTS:
        st.warning(f"Limite de {MAX_CONTRACTS} contratos. Foram enviados {len(files)}.")

    if st.button("Indexar base", type="primary", disabled=not files):
        with st.spinner("Extraindo, gerando embeddings e indexando..."):
            all_chunks: List[Chunk] = []
            names = []
            for f in files[:MAX_CONTRACTS]:
                pages = extract_pages(f)
                all_chunks.extend(chunk_pages(f.name, pages))
                names.append(f.name)
            st.session_state.vector_store = build_vector_store(client, all_chunks)
            st.session_state.indexed_contracts = names
        st.success(f"Base indexada: {len(names)} contrato(s), "
                   f"{len(st.session_state.vector_store.chunks)} blocos.")

    if st.session_state.indexed_contracts:
        with st.expander("Contratos na base", expanded=False):
            for n in st.session_state.indexed_contracts:
                st.write(f"• {n}")

    store = st.session_state.vector_store

    # Histórico
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ex.: Qual contrato possui multa superior a R$ 50.000?")
    if question:
        if store.is_empty:
            st.error("Indexe ao menos um contrato antes de perguntar.")
            return

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Consultando a base..."):
                ans = answer_question(client, store, question)

            st.markdown(ans.get("resposta", ""))
            meta = (
                f"**Contrato:** {ans.get('contrato', 'N/A')} | "
                f"**Página:** {ans.get('pagina', 'N/A')} | "
                f"**Confiança:** {ans.get('confianca', 'N/A')}"
            )
            st.caption(meta)
            if ans.get("trecho"):
                with st.expander("Trecho encontrado"):
                    st.write(ans["trecho"])
            with st.expander("Fontes recuperadas"):
                for h in ans.get("_fontes", []):
                    st.markdown(f"**{h['contract']}** — p.{h['page']} "
                                f"(score {h['score']:.2f})")
                    st.caption(h["text"][:300] + "...")

        rendered = ans.get("resposta", "") + "\n\n" + meta
        st.session_state.chat_history.append({"role": "assistant", "content": rendered})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_state()

    st.sidebar.title("⚖️ LegalGlass AI")
    st.sidebar.caption("Inteligência Artificial para Gestão Contratual e Análise Jurídica")

    client = get_client()
    if client is None:
        st.sidebar.error("Chave OpenAI não configurada.")
        st.title("⚖️ LegalGlass AI")
        st.warning(
            "Defina a chave da OpenAI em **Streamlit Secrets** para usar a aplicação:\n\n"
            "```toml\nOPENAI_API_KEY = \"sk-...\"\n```"
        )
        st.stop()

    page = st.sidebar.radio(
        "Módulos",
        ["Comparador de Contratos", "Chat Jurídico Inteligente"],
    )
    st.sidebar.divider()
    st.sidebar.caption(f"Modelo: `{CHAT_MODEL}`")
    st.sidebar.caption(f"Embeddings: `{EMBEDDING_MODEL}`")

    if page == "Comparador de Contratos":
        render_comparator(client)
    else:
        render_chat(client)


if __name__ == "__main__":
    main()
