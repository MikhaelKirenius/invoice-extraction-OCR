import os
import json
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Invoice Extractor", page_icon="🧾", layout="centered")
st.title("🧾 Invoice Extractor (OCR + NER)")

structured = {}

files_sel = st.file_uploader(
    'Upload Multiple Invoice Image(s) (JPG/PNG/JPEG*)', 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

run_btn = st.button("Process", type="primary", use_container_width=True)

if "extracted_rows" not in st.session_state:
    st.session_state.extracted_rows = []


columns = ["INVOICE_NUMBER", "INVOICE_DATE", "SELLER_NAME", "CLIENT_NAME", "TOTAL"]
row = {}

if run_btn:
    if not files_sel:
        st.warning("Choose at least one file.")
        st.stop()

    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
        if not health.ok:
            st.error(f"Health check gagal: {health.status_code} - {health.text}")
            st.stop()

        endpoint = ""

        if len(files_sel) == 1:
            endpoint = "/predict-image"
            files = {"file": (files_sel[0].name, files_sel[0].getvalue(), files_sel[0].type or "application/octet-stream")}
        else:
            endpoint = "/predict-images"
            files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files_sel]
        
        with st.spinner("Memproses di server..."):
            resp = requests.post(f"{API_URL}{endpoint}", files=files, timeout=120)

        if not resp.ok:
            st.error(f"API error: {resp.status_code}\n{resp.text}")
            st.stop()

        data = resp.json()

        st.subheader("✅ Structured Fields")

        results = []

        if "results" in data:            
            results = data["results"]
        else:                            
            results = [{
                "filename": files_sel[0].name,
                "structured": data.get("structured", {}) or {},
                "meta": data.get("meta", {}) or {}
            }]

        added_count = 0
        invoice_numbers = {r.get("INVOICE_NUMBER", "") for r in st.session_state.extracted_rows}

        for item in results:
            err = item.get("error")
            if err:
                st.warning(f"⚠️ {item.get('filename')}: {err}")
                continue

            structured = item.get("structured", {}) or {}
            if not structured:
                st.info(f"{item.get('filename')}: Tidak ada field terdeteksi.")
                continue

            # Tampilkan ringkas per file
            st.write(f"**{item.get('filename')}**")
            for k, v in structured.items():
                st.write(f"- **{k}**: {v}")

            row = {col: structured.get(col, "") for col in columns}
            inv_no = row.get("INVOICE_NUMBER", "")
            if inv_no and inv_no in invoice_numbers:
                st.warning(f"Invoice Number '{inv_no}' sudah pernah diinput.")
            else:
                st.session_state.extracted_rows.append(row)
                if inv_no:
                    invoice_numbers.add(inv_no)
                added_count += 1

        if added_count:
            st.success(f"{added_count} row(s) ditambahkan ke tabel.")

    except requests.exceptions.ConnectionError:
        st.error("Tidak bisa terhubung ke FastAPI. Pastikan server berjalan dan alamat benar.")
    except requests.exceptions.ReadTimeout:
        st.error("Request timeout. Coba lagi atau kecilkan ukuran gambar.")
    except Exception as e:
        st.exception(e)


st.subheader("📝 Full Extracted Data (CSV)")

csv_file = st.file_uploader("Upload Existing CSV (optional)", type=["csv"], key="csv_uploader")
if csv_file is not None:
    try:
        df_csv = pd.read_csv(csv_file)
        if all(col in df_csv.columns for col in columns):
            existing_invoices = {r["INVOICE_NUMBER"] for r in st.session_state.extracted_rows}
            new_rows = df_csv.to_dict(orient="records")
            added = 0
            for row in new_rows:
                if row["INVOICE_NUMBER"] not in existing_invoices:
                    st.session_state.extracted_rows.append(row)
                    existing_invoices.add(row["INVOICE_NUMBER"])
                    added += 1
            st.success(f"{added} row(s) ditambahkan dari CSV.")
        else:
            st.error("Kolom CSV tidak sesuai. Pastikan header: " + ", ".join(columns))
    except Exception as e:
        st.error(f"CSV error: {e}")



df = pd.DataFrame(st.session_state.extracted_rows, columns=columns)

st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="extracted_data.csv",
    mime="text/csv",
    use_container_width=True
)

