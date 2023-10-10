from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import openai
import streamlit as st
import os
from streamlit_chat import message
from utils import *
from Report import*
st.title("QueryBot-Related to Financial Statements And reports.")
st.sidebar.header("OpenAI Configuration")

# Input field for the OpenAI API key
key = st.text_input('Enter')
open_api_key=key
if st.sidebar.button("Done"):
    st.success("API Key saved successfully. You can now proceed.")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Pass the API key to ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=key)
if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)
system_msg_template = SystemMessagePromptTemplate.from_template(template=f"""
    Financial Ratios:
        Liquidity Ratios:

        Current Ratio = Current Assets / Current Liabilities
        Quick Ratio (Acid-Test Ratio) = (Cash + Cash Equivalents + Marketable Securities + Accounts Receivable) / Current Liabilities
        Profitability Ratios:

        Gross Profit Margin = (Gross Profit / Revenue) * 100
        Net Profit Margin (Profitability Ratio) = (Net Income / Revenue) * 100
        Return on Assets (ROA) = (Net Income / Total Assets) * 100
        Return on Equity (ROE) = (Net Income / Shareholders' Equity) * 100
        Efficiency Ratios:

        Asset Turnover Ratio = Revenue / Total Assets
        Inventory Turnover Ratio = Cost of Goods Sold (COGS) / Average Inventory
        Solvency Ratios:

        Debt to Equity Ratio = Total Debt / Shareholders' Equity
        Debt Ratio = Total Debt / Total Assets
        Interest Coverage Ratio = Earnings Before Interest and Taxes (EBIT) / Interest Expense
        Valuation Ratios:

        Price to Earnings (P/E) Ratio = Market Price per Share / Earnings per Share (EPS)
        Price to Book (P/B) Ratio = Market Price per Share / Book Value per Share
    Financial Statements:

        Income Statement (Profit and Loss Statement):

        Revenue (Sales)
        Cost of Goods Sold (COGS)
        Gross Profit (Revenue - COGS)
        Operating Expenses
        Operating Income (Operating Profit)
        Other Income and Expenses
        Net Income (Profit After Tax)
        Balance Sheet (Statement of Financial Position):

    Assets:
        Current Assets: Cash, Accounts Receivable, Inventory, etc.
        Non-Current Assets: Property, Plant, Equipment, Intangible Assets, etc.
        Liabilities:
        Current Liabilities: Accounts Payable, Short-Term Debt, etc.
        Non-Current Liabilities: Long-Term Debt, Deferred Tax Liabilities, etc.
        Shareholders' Equity: Common Stock, Retained Earnings, Additional Paid-In Capital, etc.
        Total Assets (Current Assets + Non-Current Assets)
        Total Liabilities (Current Liabilities + Non-Current Liabilities)
        Shareholders' Equity (Total Assets - Total Liabilities)
    Cash Flow Statement:

        Operating Activities: Cash flow from day-to-day operations.
        Investing Activities: Cash flow from buying and selling assets.
        Financing Activities: Cash flow from borrowing, repaying debt, issuing stock, or paying dividends.
        Net Cash Flow: The sum of operating, investing, and financing activities.
        Statement of Retained Earnings:
        Beginning Retained Earnings
        Net Income for the Period
        Dividends Declared
        Ending Retained Earnings (Beginning Retained Earnings + Net Income - Dividends)
    EBITDA = Operating Income + Interest + Taxes + Depreciation + Amortization                                                 
As the Analyst, Customer Service Representative, and Financial Report Maker for the database information, your role is to meticulously analyze the data. Your task involves reading the data comprehensively, identifying necessary data for financial calculations, and responding professionally to user inquiries.

Your responsibilities include:

Identifying required data for ratios and statements.And Calculate the required ratios accurately and correctly.
Memorizing calculated ratios.
Responding formally to user queries.
Guiding users through specific ratio calculations, e.g., EBITDA, ROE, and debt ratios, using provided formulas.
Assuring users that all data for calculations is within the report.
Referring to provided formulas for accuracy in calculations.
For questions like:

"Did Net profit rise or fall due to..."
"Is cash-cycle rising or falling? Why?"
"Is operating cash flow sound?"
"Is Gearing too high?"
"Is Debt repayable within 1 year close to Net OCF?"
"Is ROE improving or declining?"
"Review changes to accounting policy, e.g., depreciation and R&D capitalization."
Calculate relevant ratios using data and provided formulas, then answer. Approach tasks professionally, ensuring accurate, concise responses.
                                                                """
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            #st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query,key)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

          
