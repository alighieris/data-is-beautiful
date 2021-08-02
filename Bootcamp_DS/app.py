import pandas as pd
import streamlit as st
from pycaret.regression import *

#load trained model
model = load_model('model/modelo_final_cat')

# carregar uma amostra de dados
dataset = pd.read_csv('data/dataset.csv')

st.title("Data App - Predição de valores de Aluguéis")

st.markdown("Este é um Data App utilizado para exibir a solução do case, a predição do valor de aluguel de acordo com as características do imóvel.")

st.sidebar.subheader("Defina os atributos do imóvel para predição do valor de aluguel:")

# mapear dados do usuário para cada atributo

area = st.sidebar.number_input("Área total")
num_quartos = st.sidebar.number_input("Número de quartos")
num_banheiros = st.sidebar.number_input("Número de banheiros")
num_garagem = st.sidebar.number_input("Vagas de garagem")
num_andares = st.sidebar.number_input("Número de andares")
aceita_animais = st.sidebar.selectbox("Aceita animais?",('Sim','Não'))
mobilia = st.sidebar.selectbox("Mobiliado?",('Sim','Não'))

# transformar inputs em valores binários

aceita_animais = 1 if aceita_animais=='Sim' else 0
mobilia = 1 if mobilia == 'Sim' else 0

estados = { 'São Paulo': 'SP',
            'Campinas': 'SP',
            'Rio de Janeiro': 'RJ',
            'Belo Horizonte': 'MG',
            'Porto Alegre': 'RS'}

cidade = st.sidebar.selectbox("Cidade",('São Paulo',
                                        'Rio de Janeiro',
                                        'Belo Horizonte',
                                        'Campinas',
                                        'Porto Alegre'))
estado = estados[cidade]

valor_condominio = st.sidebar.number_input("Valor do condomínio")
valor_iptu = st.sidebar.number_input("Valor do IPTU")
valor_seguro_incendio = st.sidebar.number_input("Valor do seguro de incêndio")

# botão

btn_predict = st.sidebar.button("Calcular preço")

if btn_predict:
    df_test = pd.DataFrame()
    df_test['cidade'] = [cidade]
    df_test['estado'] = [estado]
    df_test['area'] = [area]
    df_test['num_quartos'] = [num_quartos]
    df_test['num_banheiros'] = [num_banheiros]
    df_test['garagem'] = [num_garagem]
    df_test['num_andares'] = [num_andares]
    df_test['aceita_animais'] = [aceita_animais]
    df_test['mobilia'] = [mobilia]
    df_test['valor_condominio'] = [valor_condominio]
    df_test['valor_iptu'] = [valor_iptu]
    df_test['valor_seguro_incendio'] = [valor_seguro_incendio]

    print(df_test)

    result = predict_model( model,
                        data=df_test)['Label']

    st.subheader("O valor de aluguel previsto para o imóvel é de R$: {:.2f}".format(result[0]))

    #st.write(result)