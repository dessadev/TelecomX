import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_json('TelecomX_Data.json')

df_normalizado = pd.json_normalize(df.to_dict('records'), sep='_')

df_sem_id = df_normalizado.drop('customerID', axis=1)

df_sem_id['Churn'] = df_sem_id['Churn'].apply(lambda x: x if x in ['Yes', 'No'] else pd.NA)
df_tratado = df_sem_id.dropna(subset=['Churn'])

df_tratado['account_Charges_Total'] = pd.to_numeric(df_tratado['account_Charges_Total'], errors='coerce')
media_total_charges = df_tratado['account_Charges_Total'].mean()
df_tratado['account_Charges_Total'].fillna(media_total_charges, inplace=True)

df_tratado['customer_SeniorCitizen'] = df_tratado['customer_SeniorCitizen'].map({0: 'No', 1: 'Yes'})

sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df_tratado, palette='pastel')
plt.title('Distribuição de Evasão de Clientes (Churn)')
plt.show()

colunas_analise = [
    'customer_gender', 'customer_SeniorCitizen', 'customer_Partner',
    'customer_Dependents', 'phone_PhoneService', 'phone_MultipleLines',
    'internet_InternetService', 'internet_OnlineSecurity', 'internet_OnlineBackup',
    'internet_DeviceProtection', 'internet_TechSupport', 'internet_StreamingTV',
    'internet_StreamingMovies', 'account_Contract', 'account_PaperlessBilling',
    'account_PaymentMethod'
]

for coluna in colunas_analise:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=coluna, hue='Churn', data=df_tratado, palette='viridis')
    plt.title(f'Distribuição de Churn por {coluna}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

colunas_numericas = ['customer_tenure', 'account_Charges_Monthly', 'account_Charges_Total']

for coluna in colunas_numericas:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y=coluna, data=df_tratado, palette='coolwarm')
    plt.title(f'Distribuição de {coluna} por Churn')
    plt.tight_layout()
    plt.show()

print("\nAnálise exploratória concluída e gráficos gerados.")