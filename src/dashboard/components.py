from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class KpiCard:
    label: str
    value: str
    help_text: str


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        .capstone-subtitle {
            color: #4b647d;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }
        .capstone-card {
            background: linear-gradient(180deg, #f8fbff 0%, #edf4fb 100%);
            border: 1px solid rgba(34, 73, 111, 0.12);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            min-height: 120px;
        }
        .capstone-card-label {
            color: #5a6b7d;
            font-size: 0.85rem;
            margin-bottom: 0.35rem;
        }
        .capstone-card-value {
            color: #0a2440;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.45rem;
        }
        .capstone-card-help {
            color: #516273;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<div class="capstone-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_kpi_cards(cards: list[KpiCard]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            st.markdown(
                (
                    '<div class="capstone-card">'
                    f'<div class="capstone-card-label">{card.label}</div>'
                    f'<div class="capstone-card-value">{card.value}</div>'
                    f'<div class="capstone-card-help">{card.help_text}</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
