{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d64de8e1-d519-405c-9374-e6bec3efb4f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\lavan\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load the model and scaler\n",
    "try:\n",
    "    model = joblib.load('placement_model.pkl')\n",
    "    scaler = joblib.load('scaler.pkl')\n",
    "except Exception as e:\n",
    "    st.error(f\"Error loading model or scaler: {e}\")\n",
    "    st.stop()\n",
    "\n",
    "st.title(\"Student Placement Prediction App\")\n",
    "\n",
    "st.markdown(\"Fill in the student details to predict if they will be placed.\")\n",
    "\n",
    "# Input sliders\n",
    "iq = st.slider(\"IQ\", 40, 160, 100)\n",
    "prev_sem_result = st.slider(\"Previous Semester Result (out of 10)\", 5.0, 10.0, 7.5)\n",
    "cgpa = st.slider(\"CGPA\", 6.0, 10.0, 7.5)\n",
    "academic_perf = st.slider(\"Academic Performance (out of 10)\", 1, 10, 5)\n",
    "internship = st.selectbox(\"Internship Experience\", ['Yes', 'No'])\n",
    "extra_score = st.slider(\"Extra Curricular Score (out of 10)\", 0, 10, 5)\n",
    "comm_skills = st.slider(\"Communication Skills (out of 10)\", 1, 10, 5)\n",
    "projects = st.slider(\"Projects Completed\", 0, 5, 2)\n",
    "\n",
    "# Encode input\n",
    "internship_encoded = 1 if internship == 'Yes' else 0\n",
    "\n",
    "input_data = np.array([[iq, prev_sem_result, cgpa, academic_perf,\n",
    "                        extra_score, comm_skills, projects, internship_encoded]])\n",
    "\n",
    "try:\n",
    "    scaled_input = scaler.transform(input_data)\n",
    "except Exception as e:\n",
    "    st.error(f\"Error during input scaling: {e}\")\n",
    "    st.stop()\n",
    "\n",
    "# Prediction\n",
    "if st.button(\"Predict Placement\"):\n",
    "    try:\n",
    "        prediction = model.predict(scaled_input)[0]\n",
    "        result = \"Placed\" if prediction == 1 else \"Not Placed\"\n",
    "        st.success(f\"The student is likely to be: **{result}**\")\n",
    "    except Exception as e:\n",
    "        st.error(f\"Prediction error: {e}\")\n",
    "\n",
    "# Feature importance (optional)\n",
    "if st.checkbox(\"Show Feature Importance\"):\n",
    "    try:\n",
    "        importances = model.feature_importances_\n",
    "        features = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance',\n",
    "                    'Extra_Curricular_Score', 'Communication_Skills',\n",
    "                    'Projects_Completed', 'Internship_Experience_Yes']\n",
    "        sorted_idx = np.argsort(importances)[::-1]\n",
    "\n",
    "        st.subheader(\"Feature Importance\")\n",
    "        fig, ax = plt.subplots()\n",
    "        ax.barh(np.array(features)[sorted_idx], importances[sorted_idx])\n",
    "        ax.set_xlabel(\"Importance\")\n",
    "        ax.invert_yaxis()\n",
    "        st.pyplot(fig)\n",
    "    except AttributeError:\n",
    "        st.warning(\"This model does not support feature importance.\")\n",
    "    except Exception as e:\n",
    "        st.error(f\"Error showing feature importance: {e}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "553b6adb-e227-47ed-ae57-f0c9f8e10e2b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
