
# Imports
import pandas            as pd
import streamlit         as st

from io                     import BytesIO
from pycaret.classification import load_model, predict_model


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'PyCaret', \
        layout="wide",
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.write("""## É utilizado o PyCaret para gerar o modelo! """)
    st.markdown("---")
    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type = ['csv','ftr'])

    # Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        df_credit = pd.read_feather(data_file_1)
        df_credit = df_credit.sample(50000)
        st.write('Arquivo utilizado:')
        st.dataframe(df_credit)
        st.write('Fazer download do modelo no botão abaixo:')
        model_saved = load_model('best_model_projeto_final')
        predict = predict_model(model_saved, data=df_credit)

        df_xlsx = to_excel(predict)
        st.download_button(label='📥 Download',
                            data=df_xlsx ,
                            file_name= 'predict.xlsx')


if __name__ == '__main__':
	main()
    








