import os
import sys

from ToxicMl.MLmodels.Hiv import HivGNNModel
from ToxicMl.MLmodels.Lipo import LipoGCNModel
from pathlib import Path
import streamlit as st
import pandas as pd


if __name__ == "__main__":
	model = HivGNNModel(Path("saved_models/deep/HIV_ChemAttention_5-128.pt"))
	modelReg = LipoGCNModel(Path("saved_models/deep/LIPO_GCN_5-128.pt"))
	df = pd.DataFrame(columns=["SMILE", "Hiv Prediction", "Lipo Prediction"])
	st.title('ToxicML')
	uploaded_file = st.file_uploader("Choose a file")

	table = st.table(df)

	if uploaded_file is not None:
		# Can be used wherever a "file-like" object is accepted:
		dataframe = pd.read_csv(uploaded_file)
		res = []
		for el in dataframe["smiles"].values.tolist():
			res.append({
				"SMILE" : el,
				"Hiv Prediction" : model.predict(el),
				"Lipo Prediction" : modelReg.predict(el)
			})
		new_df = pd.DataFrame(res)
		table.add_rows(new_df)


