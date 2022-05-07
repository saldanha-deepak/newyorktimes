# Analysis on New York Times comments & articles


The following report outlines the data scientific approaches employed to solve pressing business and user experience concerns for the New York Times, a vast journalism and publishing company. The goal of the project was two-tiered: helping the NYT understand the demographic, engagement, and use trends to improve customer outreach and marketing activities; and designing an algorithm to help NYT automate, at least partly, the process with which they select NYT Picks: comments on articles that the NYT finds insightful and resourceful for the “general readership”. The project successfully outlined geographic trends in application usage through comments data as a key performance indicator and metric for assessment. We also identified key topics, authors, and themes that prompt most engagement: news, op-ed, and editorial content. These insights were driven by robust Exploratory Data Analysis (EDA), and were corroborated by testing data collected over a year and a half. Finally, an NLP-enabled, sentiment informed classification algorithm was defined to predict which comments would be more likely to be selected as NYT Picks. The classification algorithm has substantial predictive ability, and can be successfully implemented to reduce costs, boost revenues, and improve user experience for customers using the NYT app.

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "NYT Jan Comments & Articles.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/saldanha-deepak/newyorktimes/blob/main/NYT_Jan_Comments_%26_Articles.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Analysis of New York Times articles & comments\n",
        "![image](https://cdn.vox-cdn.com/thumbor/9bhlQH0_ZpbHvClRuAK4005tp0c=/0x83:1020x593/fit-in/1200x600/cdn.vox-cdn.com/assets/964948/nytimes-logo-hq-stock_1020.jpg)\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "WipNz-Fb4tPy"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Import\n",
        "\n",
        "Kaggle ref: https://www.kaggle.com/datasets/aashita/nyt-comments"
      ],
      "metadata": {
        "id": "U4ogndE_Bb4V"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ObkMB_m1BUAf"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import os\n",
        "from os import path\n",
        "from PIL import Image\n",
        "from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator\n",
        "import re "
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eWD8tv_FcQFk",
        "outputId": "e323a08d-f976-4a3b-d1f8-d815a1d569d5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_articles = pd.read_csv(\"/content/drive/Shareddrives/Data Science with Python/Datasets/ArticlesJan2017.csv\")\n",
        "\n",
        "df_comments = pd.read_csv(\"/content/drive/Shareddrives/Data Science with Python/Datasets/CommentsJan2017.csv\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EaL4opG_BqYT",
        "outputId": "fe7bc9a9-f0a3-4006-8b66-28c0f2700c88"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2882: DtypeWarning: Columns (14,15,31,32) have mixed types.Specify dtype option on import or set low_memory=False.\n",
            "  exec(code_obj, self.user_global_ns, self.user_ns)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_articles.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L3qCVA-MeI-9",
        "outputId": "1f0ba3ce-054b-4eda-b59a-65e6e9de3593"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(850, 16)"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Merging the two dataframes based on the similar \"articleID\" column."
      ],
      "metadata": {
        "id": "l0ewQJU_EF06"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#merge the dataframes on articleID\n",
        "df = pd.merge(df_articles,df_comments,on='articleID')"
      ],
      "metadata": {
        "id": "wTf11tWyDXve"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9TstTp0AeP8e",
        "outputId": "6fde4a2f-f387-494d-b31d-83994a886ce2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(231449, 49)"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6nstdeEUhYiE",
        "outputId": "f6dfcee7-bdda-43e5-a79f-37b61b0cd2a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "articleID                     0\n",
              "abstract                 226975\n",
              "byline                        0\n",
              "documentType                  0\n",
              "headline                      0\n",
              "keywords                      0\n",
              "multimedia                    0\n",
              "newDesk_x                     0\n",
              "printPage_x                   0\n",
              "pubDate                       0\n",
              "sectionName_x                 0\n",
              "snippet                       0\n",
              "source                        0\n",
              "typeOfMaterial_x              0\n",
              "webURL                        0\n",
              "articleWordCount_x            0\n",
              "approveDate                   0\n",
              "articleWordCount_y            0\n",
              "commentBody                   0\n",
              "commentID                     0\n",
              "commentSequence               0\n",
              "commentTitle              13407\n",
              "commentType                   0\n",
              "createDate                    0\n",
              "depth                         0\n",
              "editorsSelection              0\n",
              "inReplyTo                     0\n",
              "newDesk_y                     0\n",
              "parentID                      0\n",
              "parentUserDisplayName    175223\n",
              "permID                        0\n",
              "picURL                        0\n",
              "printPage_y                   0\n",
              "recommendations               0\n",
              "recommendedFlag          231449\n",
              "replyCount                    0\n",
              "reportAbuseFlag          231449\n",
              "sectionName_y                 0\n",
              "sharing                       0\n",
              "status                        0\n",
              "timespeople                   0\n",
              "trusted                       0\n",
              "updateDate                    0\n",
              "userDisplayName              47\n",
              "userID                        0\n",
              "userLocation                 35\n",
              "userTitle                231426\n",
              "userURL                  231447\n",
              "typeOfMaterial_y              0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#save to new csv file combined_data\n",
        "#df.to_csv(\"/content/drive/Shareddrives/Data Science with Python/Datasets/Combined_Data.csv\")\n",
        "\n",
        "#note: this csv includes all columns before any pre-processing has been comp"
      ],
      "metadata": {
        "id": "Srg4BMdgEVCC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.describe()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 393
        },
        "id": "ZxkgHdZfKLuY",
        "outputId": "001ccaac-1527-4042-8dbd-5f3796076feb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "          multimedia    printPage_x  articleWordCount_x   approveDate  \\\n",
              "count  231449.000000  231449.000000       231449.000000  2.314490e+05   \n",
              "mean        0.980639       9.113934         1221.473452  1.484876e+09   \n",
              "std         0.137789      11.031421          873.465724  1.376335e+06   \n",
              "min         0.000000       0.000000           25.000000  1.483318e+09   \n",
              "25%         1.000000       1.000000          821.000000  1.484140e+09   \n",
              "50%         1.000000       1.000000         1067.000000  1.484858e+09   \n",
              "75%         1.000000      20.000000         1409.000000  1.485530e+09   \n",
              "max         1.000000      66.000000         8784.000000  1.522277e+09   \n",
              "\n",
              "       articleWordCount_y     commentID  commentSequence    createDate  \\\n",
              "count       231449.000000  2.314490e+05     2.314490e+05  2.314490e+05   \n",
              "mean          1221.473452  2.116226e+07     2.116226e+07  1.484863e+09   \n",
              "std            873.465724  1.923215e+05     1.923215e+05  1.371201e+06   \n",
              "min             25.000000  2.096371e+07     2.096371e+07  1.483314e+09   \n",
              "25%            821.000000  2.106003e+07     2.106003e+07  1.484105e+09   \n",
              "50%           1067.000000  2.115149e+07     2.115149e+07  1.484851e+09   \n",
              "75%           1409.000000  2.125085e+07     2.125085e+07  1.485518e+09   \n",
              "max           8784.000000  2.650895e+07     2.650895e+07  1.522190e+09   \n",
              "\n",
              "               depth  editorsSelection  ...    printPage_y  recommendations  \\\n",
              "count  231449.000000     231449.000000  ...  231449.000000    231449.000000   \n",
              "mean        1.243051          0.021305  ...       9.113934        22.979373   \n",
              "std         0.429017          0.144399  ...      11.031421       123.448850   \n",
              "min         1.000000          0.000000  ...       0.000000         0.000000   \n",
              "25%         1.000000          0.000000  ...       1.000000         1.000000   \n",
              "50%         1.000000          0.000000  ...       1.000000         4.000000   \n",
              "75%         1.000000          0.000000  ...      20.000000        11.000000   \n",
              "max         4.000000          1.000000  ...      66.000000      9279.000000   \n",
              "\n",
              "       recommendedFlag     replyCount  reportAbuseFlag        sharing  \\\n",
              "count              0.0  231449.000000              0.0  231449.000000   \n",
              "mean               NaN       0.538114              NaN       0.085431   \n",
              "std                NaN       3.034708              NaN       0.279523   \n",
              "min                NaN       0.000000              NaN       0.000000   \n",
              "25%                NaN       0.000000              NaN       0.000000   \n",
              "50%                NaN       0.000000              NaN       0.000000   \n",
              "75%                NaN       0.000000              NaN       0.000000   \n",
              "max                NaN     529.000000              NaN       1.000000   \n",
              "\n",
              "         timespeople        trusted    updateDate        userID  \n",
              "count  231449.000000  231449.000000  2.314490e+05  2.314490e+05  \n",
              "mean        0.998086       0.035934  1.484877e+09  4.913953e+07  \n",
              "std         0.043708       0.186127  1.377001e+06  2.301359e+07  \n",
              "min         0.000000       0.000000  1.483318e+09  1.166000e+03  \n",
              "25%         1.000000       0.000000  1.484141e+09  3.206964e+07  \n",
              "50%         1.000000       0.000000  1.484858e+09  5.699065e+07  \n",
              "75%         1.000000       0.000000  1.485531e+09  6.708296e+07  \n",
              "max         1.000000       1.000000  1.522277e+09  8.543142e+07  \n",
              "\n",
              "[8 rows x 22 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-29f4dda4-1a41-4f2e-96af-1d5e81757339\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>multimedia</th>\n",
              "      <th>printPage_x</th>\n",
              "      <th>articleWordCount_x</th>\n",
              "      <th>approveDate</th>\n",
              "      <th>articleWordCount_y</th>\n",
              "      <th>commentID</th>\n",
              "      <th>commentSequence</th>\n",
              "      <th>createDate</th>\n",
              "      <th>depth</th>\n",
              "      <th>editorsSelection</th>\n",
              "      <th>...</th>\n",
              "      <th>printPage_y</th>\n",
              "      <th>recommendations</th>\n",
              "      <th>recommendedFlag</th>\n",
              "      <th>replyCount</th>\n",
              "      <th>reportAbuseFlag</th>\n",
              "      <th>sharing</th>\n",
              "      <th>timespeople</th>\n",
              "      <th>trusted</th>\n",
              "      <th>updateDate</th>\n",
              "      <th>userID</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>231449.000000</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>2.314490e+05</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>2.314490e+05</td>\n",
              "      <td>2.314490e+05</td>\n",
              "      <td>2.314490e+05</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>...</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>0.0</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>231449.000000</td>\n",
              "      <td>2.314490e+05</td>\n",
              "      <td>2.314490e+05</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>0.980639</td>\n",
              "      <td>9.113934</td>\n",
              "      <td>1221.473452</td>\n",
              "      <td>1.484876e+09</td>\n",
              "      <td>1221.473452</td>\n",
              "      <td>2.116226e+07</td>\n",
              "      <td>2.116226e+07</td>\n",
              "      <td>1.484863e+09</td>\n",
              "      <td>1.243051</td>\n",
              "      <td>0.021305</td>\n",
              "      <td>...</td>\n",
              "      <td>9.113934</td>\n",
              "      <td>22.979373</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.538114</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.085431</td>\n",
              "      <td>0.998086</td>\n",
              "      <td>0.035934</td>\n",
              "      <td>1.484877e+09</td>\n",
              "      <td>4.913953e+07</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>0.137789</td>\n",
              "      <td>11.031421</td>\n",
              "      <td>873.465724</td>\n",
              "      <td>1.376335e+06</td>\n",
              "      <td>873.465724</td>\n",
              "      <td>1.923215e+05</td>\n",
              "      <td>1.923215e+05</td>\n",
              "      <td>1.371201e+06</td>\n",
              "      <td>0.429017</td>\n",
              "      <td>0.144399</td>\n",
              "      <td>...</td>\n",
              "      <td>11.031421</td>\n",
              "      <td>123.448850</td>\n",
              "      <td>NaN</td>\n",
              "      <td>3.034708</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.279523</td>\n",
              "      <td>0.043708</td>\n",
              "      <td>0.186127</td>\n",
              "      <td>1.377001e+06</td>\n",
              "      <td>2.301359e+07</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>25.000000</td>\n",
              "      <td>1.483318e+09</td>\n",
              "      <td>25.000000</td>\n",
              "      <td>2.096371e+07</td>\n",
              "      <td>2.096371e+07</td>\n",
              "      <td>1.483314e+09</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>...</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.483318e+09</td>\n",
              "      <td>1.166000e+03</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>821.000000</td>\n",
              "      <td>1.484140e+09</td>\n",
              "      <td>821.000000</td>\n",
              "      <td>2.106003e+07</td>\n",
              "      <td>2.106003e+07</td>\n",
              "      <td>1.484105e+09</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>...</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.484141e+09</td>\n",
              "      <td>3.206964e+07</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1067.000000</td>\n",
              "      <td>1.484858e+09</td>\n",
              "      <td>1067.000000</td>\n",
              "      <td>2.115149e+07</td>\n",
              "      <td>2.115149e+07</td>\n",
              "      <td>1.484851e+09</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>...</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.484858e+09</td>\n",
              "      <td>5.699065e+07</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>20.000000</td>\n",
              "      <td>1409.000000</td>\n",
              "      <td>1.485530e+09</td>\n",
              "      <td>1409.000000</td>\n",
              "      <td>2.125085e+07</td>\n",
              "      <td>2.125085e+07</td>\n",
              "      <td>1.485518e+09</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>...</td>\n",
              "      <td>20.000000</td>\n",
              "      <td>11.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.485531e+09</td>\n",
              "      <td>6.708296e+07</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>1.000000</td>\n",
              "      <td>66.000000</td>\n",
              "      <td>8784.000000</td>\n",
              "      <td>1.522277e+09</td>\n",
              "      <td>8784.000000</td>\n",
              "      <td>2.650895e+07</td>\n",
              "      <td>2.650895e+07</td>\n",
              "      <td>1.522190e+09</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>...</td>\n",
              "      <td>66.000000</td>\n",
              "      <td>9279.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>529.000000</td>\n",
              "      <td>NaN</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.522277e+09</td>\n",
              "      <td>8.543142e+07</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>8 rows × 22 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-29f4dda4-1a41-4f2e-96af-1d5e81757339')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-29f4dda4-1a41-4f2e-96af-1d5e81757339 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-29f4dda4-1a41-4f2e-96af-1d5e81757339');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "list(df.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XD_OKxlnD6QT",
        "outputId": "adc66f37-6a93-45f7-f082-fd6fee22b0fd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['articleID',\n",
              " 'abstract',\n",
              " 'byline',\n",
              " 'documentType',\n",
              " 'headline',\n",
              " 'keywords',\n",
              " 'multimedia',\n",
              " 'newDesk_x',\n",
              " 'printPage_x',\n",
              " 'pubDate',\n",
              " 'sectionName_x',\n",
              " 'snippet',\n",
              " 'source',\n",
              " 'typeOfMaterial_x',\n",
              " 'webURL',\n",
              " 'articleWordCount_x',\n",
              " 'approveDate',\n",
              " 'articleWordCount_y',\n",
              " 'commentBody',\n",
              " 'commentID',\n",
              " 'commentSequence',\n",
              " 'commentTitle',\n",
              " 'commentType',\n",
              " 'createDate',\n",
              " 'depth',\n",
              " 'editorsSelection',\n",
              " 'inReplyTo',\n",
              " 'newDesk_y',\n",
              " 'parentID',\n",
              " 'parentUserDisplayName',\n",
              " 'permID',\n",
              " 'picURL',\n",
              " 'printPage_y',\n",
              " 'recommendations',\n",
              " 'recommendedFlag',\n",
              " 'replyCount',\n",
              " 'reportAbuseFlag',\n",
              " 'sectionName_y',\n",
              " 'sharing',\n",
              " 'status',\n",
              " 'timespeople',\n",
              " 'trusted',\n",
              " 'updateDate',\n",
              " 'userDisplayName',\n",
              " 'userID',\n",
              " 'userLocation',\n",
              " 'userTitle',\n",
              " 'userURL',\n",
              " 'typeOfMaterial_y']"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data Pre-Processing"
      ],
      "metadata": {
        "id": "cO2EfWpqHY5r"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Dropping Unnecessary Columns"
      ],
      "metadata": {
        "id": "pCgmJVOvNG-2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#need the column editorsSelection & recommendedFlag\n",
        "df.drop([\"abstract\",\"documentType\",\"multimedia\",\"printPage_y\",\"source\",\"commentTitle\",\"reportAbuseFlag\",\"timespeople\",\"trusted\",\"updateDate\",\"userTitle\",\"userURL\"],axis = 1, inplace = True)"
      ],
      "metadata": {
        "id": "KFTQbV82IYYg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.drop(\"printPage_x\", axis = 1, inplace = True)"
      ],
      "metadata": {
        "id": "HZL5bVTnRcBF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "list(df.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p-ia2zdeR-Rn",
        "outputId": "e7f1ce5a-97ae-4ab8-cd52-54345b49ac35"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['articleID',\n",
              " 'byline',\n",
              " 'headline',\n",
              " 'keywords',\n",
              " 'newDesk_x',\n",
              " 'pubDate',\n",
              " 'sectionName_x',\n",
              " 'snippet',\n",
              " 'typeOfMaterial_x',\n",
              " 'webURL',\n",
              " 'articleWordCount_x',\n",
              " 'approveDate',\n",
              " 'articleWordCount_y',\n",
              " 'commentBody',\n",
              " 'commentID',\n",
              " 'commentSequence',\n",
              " 'commentType',\n",
              " 'createDate',\n",
              " 'depth',\n",
              " 'editorsSelection',\n",
              " 'inReplyTo',\n",
              " 'newDesk_y',\n",
              " 'parentID',\n",
              " 'parentUserDisplayName',\n",
              " 'permID',\n",
              " 'picURL',\n",
              " 'recommendations',\n",
              " 'recommendedFlag',\n",
              " 'replyCount',\n",
              " 'sectionName_y',\n",
              " 'sharing',\n",
              " 'status',\n",
              " 'userDisplayName',\n",
              " 'userID',\n",
              " 'userLocation',\n",
              " 'typeOfMaterial_y']"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 733
        },
        "id": "09_T6LBoQ_vh",
        "outputId": "4577a5f8-a1ba-4b21-c7f4-769b15a4419b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                  articleID                  byline  \\\n",
              "0  58691a5795d0e039260788b9  By JENNIFER STEINHAUER   \n",
              "1  58691a5795d0e039260788b9  By JENNIFER STEINHAUER   \n",
              "2  58691a5795d0e039260788b9  By JENNIFER STEINHAUER   \n",
              "3  58691a5795d0e039260788b9  By JENNIFER STEINHAUER   \n",
              "4  58691a5795d0e039260788b9  By JENNIFER STEINHAUER   \n",
              "\n",
              "                                            headline  \\\n",
              "0   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "1   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "2   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "3   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "4   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "\n",
              "                                            keywords newDesk_x  \\\n",
              "0  ['United States Politics and Government', 'Law...  National   \n",
              "1  ['United States Politics and Government', 'Law...  National   \n",
              "2  ['United States Politics and Government', 'Law...  National   \n",
              "3  ['United States Politics and Government', 'Law...  National   \n",
              "4  ['United States Politics and Government', 'Law...  National   \n",
              "\n",
              "               pubDate sectionName_x  \\\n",
              "0  2017-01-01 15:03:38      Politics   \n",
              "1  2017-01-01 15:03:38      Politics   \n",
              "2  2017-01-01 15:03:38      Politics   \n",
              "3  2017-01-01 15:03:38      Politics   \n",
              "4  2017-01-01 15:03:38      Politics   \n",
              "\n",
              "                                             snippet typeOfMaterial_x  \\\n",
              "0  The most powerful and ambitious Republican-led...             News   \n",
              "1  The most powerful and ambitious Republican-led...             News   \n",
              "2  The most powerful and ambitious Republican-led...             News   \n",
              "3  The most powerful and ambitious Republican-led...             News   \n",
              "4  The most powerful and ambitious Republican-led...             News   \n",
              "\n",
              "                                              webURL  ...  recommendations  \\\n",
              "0  https://www.nytimes.com/2017/01/01/us/politics...  ...                5   \n",
              "1  https://www.nytimes.com/2017/01/01/us/politics...  ...                3   \n",
              "2  https://www.nytimes.com/2017/01/01/us/politics...  ...                3   \n",
              "3  https://www.nytimes.com/2017/01/01/us/politics...  ...                3   \n",
              "4  https://www.nytimes.com/2017/01/01/us/politics...  ...                3   \n",
              "\n",
              "   recommendedFlag  replyCount sectionName_y  sharing    status  \\\n",
              "0              NaN           0      Politics        0  approved   \n",
              "1              NaN           0      Politics        0  approved   \n",
              "2              NaN           0      Politics        0  approved   \n",
              "3              NaN           0      Politics        0  approved   \n",
              "4              NaN           0      Politics        0  approved   \n",
              "\n",
              "  userDisplayName      userID   userLocation  typeOfMaterial_y  \n",
              "0        N. Smith  64679318.0  New York City              News  \n",
              "1     Kilocharlie  69254188.0        Phoenix              News  \n",
              "2     Frank Fryer  76788711.0        Florida              News  \n",
              "3     James Young  72718862.0        Seattle              News  \n",
              "4              M.   7529267.0        Seattle              News  \n",
              "\n",
              "[5 rows x 36 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-36a307db-9cc5-4005-b675-d9152d778bd0\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>articleID</th>\n",
              "      <th>byline</th>\n",
              "      <th>headline</th>\n",
              "      <th>keywords</th>\n",
              "      <th>newDesk_x</th>\n",
              "      <th>pubDate</th>\n",
              "      <th>sectionName_x</th>\n",
              "      <th>snippet</th>\n",
              "      <th>typeOfMaterial_x</th>\n",
              "      <th>webURL</th>\n",
              "      <th>...</th>\n",
              "      <th>recommendations</th>\n",
              "      <th>recommendedFlag</th>\n",
              "      <th>replyCount</th>\n",
              "      <th>sectionName_y</th>\n",
              "      <th>sharing</th>\n",
              "      <th>status</th>\n",
              "      <th>userDisplayName</th>\n",
              "      <th>userID</th>\n",
              "      <th>userLocation</th>\n",
              "      <th>typeOfMaterial_y</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>By JENNIFER STEINHAUER</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>5</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>N. Smith</td>\n",
              "      <td>64679318.0</td>\n",
              "      <td>New York City</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>By JENNIFER STEINHAUER</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>Kilocharlie</td>\n",
              "      <td>69254188.0</td>\n",
              "      <td>Phoenix</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>By JENNIFER STEINHAUER</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>Frank Fryer</td>\n",
              "      <td>76788711.0</td>\n",
              "      <td>Florida</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>By JENNIFER STEINHAUER</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>James Young</td>\n",
              "      <td>72718862.0</td>\n",
              "      <td>Seattle</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>By JENNIFER STEINHAUER</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>M.</td>\n",
              "      <td>7529267.0</td>\n",
              "      <td>Seattle</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 36 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-36a307db-9cc5-4005-b675-d9152d778bd0')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-36a307db-9cc5-4005-b675-d9152d778bd0 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-36a307db-9cc5-4005-b675-d9152d778bd0');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Checking for Missing Values"
      ],
      "metadata": {
        "id": "GP6slEwuTJWa"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "g5xiP-Q8S0hh",
        "outputId": "b1cd740f-9398-4215-a01b-2fdbfe67a0f8"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "articleID                     0\n",
              "byline                        0\n",
              "headline                      0\n",
              "keywords                      0\n",
              "newDesk_x                     0\n",
              "pubDate                       0\n",
              "sectionName_x                 0\n",
              "snippet                       0\n",
              "typeOfMaterial_x              0\n",
              "webURL                        0\n",
              "articleWordCount_x            0\n",
              "approveDate                   0\n",
              "articleWordCount_y            0\n",
              "commentBody                   0\n",
              "commentID                     0\n",
              "commentSequence               0\n",
              "commentType                   0\n",
              "createDate                    0\n",
              "depth                         0\n",
              "editorsSelection              0\n",
              "inReplyTo                     0\n",
              "newDesk_y                     0\n",
              "parentID                      0\n",
              "parentUserDisplayName    175223\n",
              "permID                        0\n",
              "picURL                        0\n",
              "recommendations               0\n",
              "recommendedFlag          231449\n",
              "replyCount                    0\n",
              "sectionName_y                 0\n",
              "sharing                       0\n",
              "status                        0\n",
              "userDisplayName              47\n",
              "userID                        0\n",
              "userLocation                 35\n",
              "typeOfMaterial_y              0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "There are only missing values or null values in the following columns:\n",
        "* parentUserDisplayName \n",
        "* userDisplayName\n",
        "* userLocation \n",
        "---\n",
        "* **Note:** Because these three columns are strings, we cannot impute any missing data to fill the missing data. \n"
      ],
      "metadata": {
        "id": "eQTIGZ3ZS7gx"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Checking Data Types\n",
        "\n",
        "Data Types look fine. \n",
        "\n",
        "* Object corresponds to string variables. \n",
        "* int64 corresponds to integer variables. \n",
        "* float64 corresponds to float variables. "
      ],
      "metadata": {
        "id": "-57rmwGaTnyB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.dtypes"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8QbRvwYzTcvh",
        "outputId": "e76cdc23-1dfc-4e10-c631-7fe3fa586d4b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "articleID                 object\n",
              "byline                    object\n",
              "headline                  object\n",
              "keywords                  object\n",
              "newDesk_x                 object\n",
              "pubDate                   object\n",
              "sectionName_x             object\n",
              "snippet                   object\n",
              "typeOfMaterial_x          object\n",
              "webURL                    object\n",
              "articleWordCount_x         int64\n",
              "approveDate                int64\n",
              "articleWordCount_y         int64\n",
              "commentBody               object\n",
              "commentID                float64\n",
              "commentSequence          float64\n",
              "commentType               object\n",
              "createDate                 int64\n",
              "depth                    float64\n",
              "editorsSelection           int64\n",
              "inReplyTo                  int64\n",
              "newDesk_y                 object\n",
              "parentID                 float64\n",
              "parentUserDisplayName     object\n",
              "permID                    object\n",
              "picURL                    object\n",
              "recommendations            int64\n",
              "recommendedFlag          float64\n",
              "replyCount                 int64\n",
              "sectionName_y             object\n",
              "sharing                    int64\n",
              "status                    object\n",
              "userDisplayName           object\n",
              "userID                   float64\n",
              "userLocation              object\n",
              "typeOfMaterial_y          object\n",
              "dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Cleaning The Text Data\n",
        "\n"
      ],
      "metadata": {
        "id": "HLGHUqEM0O91"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "We need to remove all punctuation from string variables to make the data more usable. "
      ],
      "metadata": {
        "id": "XUdngNzrMp_9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Cleaning up the byline column and renaming it to author\n",
        "\n",
        "df[\"byline\"] = df[\"byline\"].str.replace(\"By\", \"\")\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "TaTobg8t0Ol4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.rename(columns={'byline': 'author'},inplace =True)"
      ],
      "metadata": {
        "id": "CocuQB1U726p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Making the author column lowercase \n",
        "df['author'] = df['author'].str.lower()\n",
        "# Making the comment column lowercase \n",
        "df['commentBody'] = df['commentBody'].str.lower()"
      ],
      "metadata": {
        "id": "FD90VlS0R9E6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df1 = df.copy()\n",
        "df2 = df.copy()"
      ],
      "metadata": {
        "id": "xnnwjdlZOyyj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df2 = df2[df2['author'].str.contains('the editorial board')==False]\n",
        "\n",
        "df2 = df2[df2['commentBody'].str.contains('br')==False]"
      ],
      "metadata": {
        "id": "9UsDX2yiUbnz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 733
        },
        "id": "lNkUX6lCnSvj",
        "outputId": "1b26b650-3b48-4a2e-b71e-5d27a88d1754"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                  articleID                author  \\\n",
              "0  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "1  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "2  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "3  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "4  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "\n",
              "                                            headline  \\\n",
              "0   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "1   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "2   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "3   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "4   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "\n",
              "                                            keywords newDesk_x  \\\n",
              "0  ['United States Politics and Government', 'Law...  National   \n",
              "1  ['United States Politics and Government', 'Law...  National   \n",
              "2  ['United States Politics and Government', 'Law...  National   \n",
              "3  ['United States Politics and Government', 'Law...  National   \n",
              "4  ['United States Politics and Government', 'Law...  National   \n",
              "\n",
              "               pubDate sectionName_x  \\\n",
              "0  2017-01-01 15:03:38      Politics   \n",
              "1  2017-01-01 15:03:38      Politics   \n",
              "2  2017-01-01 15:03:38      Politics   \n",
              "3  2017-01-01 15:03:38      Politics   \n",
              "4  2017-01-01 15:03:38      Politics   \n",
              "\n",
              "                                             snippet typeOfMaterial_x  \\\n",
              "0  The most powerful and ambitious Republican-led...             News   \n",
              "1  The most powerful and ambitious Republican-led...             News   \n",
              "2  The most powerful and ambitious Republican-led...             News   \n",
              "3  The most powerful and ambitious Republican-led...             News   \n",
              "4  The most powerful and ambitious Republican-led...             News   \n",
              "\n",
              "                                              webURL  ...  recommendations  \\\n",
              "0  https://www.nytimes.com/2017/01/01/us/politics...  ...                5   \n",
              "1  https://www.nytimes.com/2017/01/01/us/politics...  ...                3   \n",
              "2  https://www.nytimes.com/2017/01/01/us/politics...  ...                3   \n",
              "3  https://www.nytimes.com/2017/01/01/us/politics...  ...                3   \n",
              "4  https://www.nytimes.com/2017/01/01/us/politics...  ...                3   \n",
              "\n",
              "   recommendedFlag  replyCount sectionName_y  sharing    status  \\\n",
              "0              NaN           0      Politics        0  approved   \n",
              "1              NaN           0      Politics        0  approved   \n",
              "2              NaN           0      Politics        0  approved   \n",
              "3              NaN           0      Politics        0  approved   \n",
              "4              NaN           0      Politics        0  approved   \n",
              "\n",
              "  userDisplayName      userID   userLocation  typeOfMaterial_y  \n",
              "0        N. Smith  64679318.0  New York City              News  \n",
              "1     Kilocharlie  69254188.0        Phoenix              News  \n",
              "2     Frank Fryer  76788711.0        Florida              News  \n",
              "3     James Young  72718862.0        Seattle              News  \n",
              "4              M.   7529267.0        Seattle              News  \n",
              "\n",
              "[5 rows x 36 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-646aecf1-3bf9-4d71-be79-08dadcdc823d\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>articleID</th>\n",
              "      <th>author</th>\n",
              "      <th>headline</th>\n",
              "      <th>keywords</th>\n",
              "      <th>newDesk_x</th>\n",
              "      <th>pubDate</th>\n",
              "      <th>sectionName_x</th>\n",
              "      <th>snippet</th>\n",
              "      <th>typeOfMaterial_x</th>\n",
              "      <th>webURL</th>\n",
              "      <th>...</th>\n",
              "      <th>recommendations</th>\n",
              "      <th>recommendedFlag</th>\n",
              "      <th>replyCount</th>\n",
              "      <th>sectionName_y</th>\n",
              "      <th>sharing</th>\n",
              "      <th>status</th>\n",
              "      <th>userDisplayName</th>\n",
              "      <th>userID</th>\n",
              "      <th>userLocation</th>\n",
              "      <th>typeOfMaterial_y</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>5</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>N. Smith</td>\n",
              "      <td>64679318.0</td>\n",
              "      <td>New York City</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>Kilocharlie</td>\n",
              "      <td>69254188.0</td>\n",
              "      <td>Phoenix</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>Frank Fryer</td>\n",
              "      <td>76788711.0</td>\n",
              "      <td>Florida</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>James Young</td>\n",
              "      <td>72718862.0</td>\n",
              "      <td>Seattle</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>3</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>M.</td>\n",
              "      <td>7529267.0</td>\n",
              "      <td>Seattle</td>\n",
              "      <td>News</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 36 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-646aecf1-3bf9-4d71-be79-08dadcdc823d')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-646aecf1-3bf9-4d71-be79-08dadcdc823d button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-646aecf1-3bf9-4d71-be79-08dadcdc823d');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Exploratory Analysis\n",
        "\n",
        "Ideas for EDA\n",
        "\n",
        "\n",
        "1.   Text classification with sklearn/Wordcloud \n",
        "2.   Popular Genre in news \n",
        "3.   Most popular Author based on comments (freq of comments) \n",
        "4.   Comment Engagement by Author \n",
        "5.   Classify the most common user locations \n",
        "6.   Determine rates of engagement of article categories in specific locations \n",
        "7.  Explore Politics/More polarized opinions for EDA\n",
        "8.  Author Specific Analysis (polarity of comments)\n",
        "9.  Determine popularity of article categories by using our own coefficient. Yakun\n",
        "10. LDA for Topic Modeling Caleb\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "H60X85EaHW79"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## EDA1: Word Cloud of Comments"
      ],
      "metadata": {
        "id": "AXpgo8Ed_GBM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "text = \" \".join(comment for comment in df.commentBody)\n",
        "stop_words = [\"br\", \"still\", \"seem\", \"said\", \"thing\",\"even\",\"u\",\"much\", \"maybe\",\"well\",\"ye\",\"say\",\"many\",\"look\",\"one\",\"now\",\"going\",\"really\",\"make\",\"want\",\"made\",\"sure\",\"may\",\"way\"] + list(STOPWORDS)\n",
        "wordcloud = WordCloud(stopwords = stop_words, max_words=100, background_color=\"white\").generate(text)\n",
        "plt.figure()\n",
        "plt.imshow(wordcloud, interpolation=\"bilinear\")\n",
        "plt.axis(\"off\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        },
        "id": "eo9ZM-k8_FHu",
        "outputId": "086ce088-b81f-4320-dba0-1337743ffb63"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# To Save the Word Cloud \n",
        "wordcloud.to_file(\"/content/drive/Shareddrives/Data Science with Python/Datasets/Comments_Word_Cloud.png\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ShhNK4kpAH6D",
        "outputId": "90ecf315-0df4-4aa6-ec81-c9721a53a09d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<wordcloud.wordcloud.WordCloud at 0x7fd7783ddf50>"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "text = \" \".join(author for author in df2.author)\n",
        "stop_words = [\"br\", \"still\", \"seem\", \"said\", \"thing\",\"even\",\"u\",\"much\", \"maybe\",\"well\",\"ye\",\"say\",\"many\",\"look\",\"one\",\"now\",\"going\",\"really\",\"make\",\"want\",\"made\",\"sure\",\"may\",\"way\"] + list(STOPWORDS)\n",
        "wordcloud = WordCloud(stopwords = stop_words, max_words=50, background_color=\"white\").generate(text)\n",
        "plt.figure()\n",
        "plt.imshow(wordcloud, interpolation=\"bilinear\")\n",
        "plt.axis(\"off\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 197
        },
        "id": "OPejm9uYtxCl",
        "outputId": "d2d16bb1-8aaa-4121-c1a1-18e8587438cd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "wordcloud.to_file(\"/content/drive/Shareddrives/Data Science with Python/Datasets/Authors_Word_Cloud.png\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PXeabM_It3HP",
        "outputId": "d9ef4124-6a9b-4d35-ddbc-8ab31479ddd2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<wordcloud.wordcloud.WordCloud at 0x7f347415e990>"
            ]
          },
          "metadata": {},
          "execution_count": 93
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **EDA2:** Popular Article Types"
      ],
      "metadata": {
        "id": "W6emR35l_MDI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.groupby('typeOfMaterial_y').commentBody.count()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3U15S8t6wMf6",
        "outputId": "1433405d-62fa-402f-9e1f-c3ea6f7440e1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "typeOfMaterial_y\n",
              "Blog                 4474\n",
              "Brief                 207\n",
              "Editorial           21869\n",
              "Letter                100\n",
              "News               121114\n",
              "News Analysis        2612\n",
              "Obituary (Obit)       497\n",
              "Op-Ed               78951\n",
              "Question              128\n",
              "Review               1045\n",
              "briefing              452\n",
              "Name: commentBody, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 63
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "fig = plt.figure(figsize=(10,10))\n",
        "df.groupby('typeOfMaterial_y').commentBody.count().sort_values(ascending=False).plot.bar(ylim=0)\n",
        "plt.title('Comments Per Article Type', fontsize = 20)\n",
        "plt.xticks(fontsize=15)\n",
        "plt.yticks(fontsize=15)\n",
        "plt.xlabel('Article Type', fontsize=15)\n",
        "plt.ylabel('Number of Comments', fontsize=15)\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 735
        },
        "id": "ddhQnLbQ2UFk",
        "outputId": "a6be19b0-5651-4d7c-e315-016e446d7180"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x720 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **EDA3:** Most Comments by Author's Articles"
      ],
      "metadata": {
        "id": "9d4JQ6y-_8Pc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# 2\n",
        "df2 = df2[df2['author'].str.contains('the editorial board')==False]\n",
        "#Removing the editorial board from the list of authors\n",
        "\n",
        "import matplotlib.pyplot as plt\n",
        "fig = plt.figure(figsize=(10,10))\n",
        "df2.groupby('author').commentBody.count()[lambda x: x >= 2000].sort_values(ascending=False).plot.bar()\n",
        "plt.title('Comments Per Author', fontsize=20)\n",
        "plt.xticks(fontsize=15)\n",
        "plt.yticks(fontsize=15)\n",
        "plt.xticks(fontsize=15)\n",
        "plt.xlabel('Author Name', fontsize=15)\n",
        "plt.ylabel('Total Number of Comments', fontsize=15)\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "tr-RS6nT7oRT",
        "outputId": "b78ee29b-4759-4d01-c98e-02e33a542fec"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x720 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Need to remove editorial board from dataset; showing up as an outlier."
      ],
      "metadata": {
        "id": "sFJGsTlaYrau"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **EDA4:** Author Engagement"
      ],
      "metadata": {
        "id": "qAVMRboOAKxn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Authors With the Least Amount of Engagement\n",
        "import matplotlib.pyplot as plt\n",
        "plt.figure(figsize=(7,7))\n",
        "df1.groupby('author').commentBody.count()[lambda x: x <= 5].sort_values(ascending=False).plot.bar()\n",
        "plt.title('Authors with 10 Comments or Less', fontsize=20)\n",
        "plt.xlabel('Author', fontsize=15)\n",
        "plt.yticks(fontsize=15)\n",
        "plt.xticks(fontsize=15)\n",
        "plt.ylabel('Number of Comments', fontsize=15)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 883
        },
        "id": "jvM0GMwKbEs2",
        "outputId": "a97103b0-f8c8-4feb-afb6-b14c73e3146d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 504x504 with 1 Axes>"
            ],
            "image/png": "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i5mNAcYAjBw5cohzIyIyvIza95+l88f9YtNapc2a7u/g3P1Ed1/Z3VceMWLEUGdHRESGiek+wImIiHRCAU5ERGpp0OvgzGwuoqM3wBLAvGa2Vfr/AncfP9h5EhGR+hmKRiaLAGc1TWv8vwwwblBzIyIitTToAc7dxwE22NsVEZEZi+rgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklhTgRESklgY9wJnZ8mZ2uZmNN7OnzOzHZjbzYOdDRETqbZbB3JiZLQBcBtwLbAYsCxxJBNofDGZeRESk3gY1wAHfBOYEtnD314BLzWxe4GAzOyxNExER6dpgF1F+Bri4KZD9mQh66wxyXkREpMYGO8B9CLgvO8HdHwPGp3kiIiI9Ye4+eBszexvYy91/1TT9CeA0d98/J80YYEz694PA/QWrXxh4ocOsKe30n3Yot620Squ002/apd19RO4cdx+0F/A2sHvO9CeAn3W57n8rbX3TDtd8K63SKu3QpR3sIsqXgflypi+Q5omIiPTEYAe4+2iqazOzpYC5aKqbExER6cZgB7gLgU+b2TyZadsAE4Cru1z3iUpb67RDuW2lVVqlHYZpB7uRyQJEJ++7gUOB9wFHAb9yd3X0FhGRnhnUAAcxVBdwDLAG8ApwEnCwu08e1IyIiEitDXqAExERGQx6msAwYGYf7TDdHGZ2iZmt20Ha2c3sADP7WCfb7mB7k81s1fT3KWa2TAfrmNXMPmlmi3eQdlD3dzjr5rgajtsdKjPg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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **EDA7**:\n",
        "\n"
      ],
      "metadata": {
        "id": "E4bul4VlQK2P"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df.dtypes"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6aAsTnvDh1mW",
        "outputId": "d6962012-8dcd-486b-e057-df0517077a76"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "articleID                 object\n",
              "author                    object\n",
              "headline                  object\n",
              "keywords                  object\n",
              "newDesk_x                 object\n",
              "pubDate                   object\n",
              "sectionName_x             object\n",
              "snippet                   object\n",
              "typeOfMaterial_x          object\n",
              "webURL                    object\n",
              "articleWordCount_x         int64\n",
              "approveDate                int64\n",
              "articleWordCount_y         int64\n",
              "commentBody               object\n",
              "commentID                float64\n",
              "commentSequence          float64\n",
              "commentType               object\n",
              "createDate                 int64\n",
              "depth                    float64\n",
              "editorsSelection           int64\n",
              "inReplyTo                  int64\n",
              "newDesk_y                 object\n",
              "parentID                 float64\n",
              "parentUserDisplayName     object\n",
              "permID                    object\n",
              "picURL                    object\n",
              "recommendations            int64\n",
              "recommendedFlag          float64\n",
              "replyCount                 int64\n",
              "sectionName_y             object\n",
              "sharing                    int64\n",
              "status                    object\n",
              "userDisplayName           object\n",
              "userID                   float64\n",
              "userLocation              object\n",
              "typeOfMaterial_y          object\n",
              "dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 69
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "fig = plt.figure(figsize=(10,10))\n",
        "df1.groupby('sectionName_x').commentBody.count().plot.bar()\n",
        "plt.title('Authors with Comments on Politics', fontsize=15)\n",
        "plt.xlabel('Article Types')\n",
        "plt.ylabel('Number of Comments')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 706
        },
        "id": "15EWXWekh0I5",
        "outputId": "e00925bf-62f9-4f8b-9d58-d020afb1edbe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x720 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df['snippet'][0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 53
        },
        "id": "CR2jPyQvoo9U",
        "outputId": "47dead88-15c1-460f-db10-3cf069a30f74"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'The most powerful and ambitious Republican-led Congress in 20 years plans quick action on several priorities — most notably to clear a path for the repeal of President Obama’s health care law.'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 90
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import nltk\n",
        "import spacy\n",
        "from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA\n"
      ],
      "metadata": {
        "id": "dFbLPXGlphNa",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2db10c95-a9e1-448b-e8b9-35c954dc3d21"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/nltk/twitter/__init__.py:20: UserWarning: The twython library has not been installed. Some functionality from the twitter package will not be available.\n",
            "  warnings.warn(\"The twython library has not been installed. \"\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "comments=df['commentBody']\n",
        "comments"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qOJVeag3v9Kj",
        "outputId": "45aeeb46-a162-484b-b800-3f72b19e1abf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0         for all you americans out there --- still rejo...\n",
              "1         obamas policies may prove to be the least of t...\n",
              "2         democrats are comprised of malcontents who gen...\n",
              "3         the picture in this article is the face of con...\n",
              "4                              elections have consequences.\n",
              "                                ...                        \n",
              "231444    so much ignorance in one post. it physically h...\n",
              "231445    @resonance<br/>we didn't know that \"fact\" beca...\n",
              "231446    @resonance<br/>he did not do the same thing \"t...\n",
              "231447    @jody - i agree - and did not mean to appear a...\n",
              "231448    @ resonance kansas - res i assume you ask abou...\n",
              "Name: commentBody, Length: 231449, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 92
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "from textblob import TextBlob\n",
        "\n",
        "import warnings \n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "import matplotlib as mpl\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from plotly.offline import init_notebook_mode, iplot\n",
        "import plotly.figure_factory as ff\n",
        "init_notebook_mode(connected=True)\n",
        "plt.style.use('fivethirtyeight')\n",
        "%matplotlib inline\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 17
        },
        "id": "xbVIcINT18hU",
        "outputId": "2f175490-ab7b-4654-eade-faca0818e7f7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "        <script type=\"text/javascript\">\n",
              "        window.PlotlyConfig = {MathJaxConfig: 'local'};\n",
              "        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}\n",
              "        if (typeof require !== 'undefined') {\n",
              "        require.undef(\"plotly\");\n",
              "        requirejs.config({\n",
              "            paths: {\n",
              "                'plotly': ['https://cdn.plot.ly/plotly-2.8.3.min']\n",
              "            }\n",
              "        });\n",
              "        require(['plotly'], function(Plotly) {\n",
              "            window._Plotly = Plotly;\n",
              "        });\n",
              "        }\n",
              "        </script>\n",
              "        "
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from textblob import TextBlob"
      ],
      "metadata": {
        "id": "UnWFfHszxX04"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "comments['sentiment'] = df['commentBody'].map(lambda text: TextBlob(text).sentiment.polarity)\n",
        "comments['sentiment']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "piSYzagco3bC",
        "outputId": "e68910ef-35e7-4388-a5d6-1365315c1d17"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0         0.362500\n",
              "1        -0.072727\n",
              "2         0.025000\n",
              "3         0.198898\n",
              "4         0.000000\n",
              "            ...   \n",
              "231444    0.050000\n",
              "231445    0.000000\n",
              "231446    0.042708\n",
              "231447    0.139534\n",
              "231448    0.000000\n",
              "Name: commentBody, Length: 231449, dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 96
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **EDA8**:"
      ],
      "metadata": {
        "id": "iAeDqhH_QRO5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "mpl.rcParams['figure.figsize'] = (18, 8)\n",
        "mpl.rcParams['axes.titlesize'] = 'xx-large'\n",
        "mpl.rcParams['axes.labelsize'] = 'x-large'\n",
        "sns.distplot(comments.sentiment);\n",
        "plt.yticks(fontsize=15)\n",
        "plt.xticks(fontsize=15)\n",
        "plt.title(\"Distribution of sentiment polarity of comments\", fontsize=20);"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 523
        },
        "id": "Nu9XWYFDxupa",
        "outputId": "2710482d-0ec0-4cba-9d21-747177be875c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1296x576 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Sentiment plays a notable role in determining popularity. People are more likely to comment on articles with headlines that have negative sentiment, and less likely to comment on articles with headlines that have neutral sentiment. Previous research has shown that content that evokes high-arousal positive (awe) or negative (anger or anxiety) emotions tends to be more viral."
      ],
      "metadata": {
        "id": "gIYbO5taeO9q"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**bold text**## **EDA9:** LDA Topic Modeling\n"
      ],
      "metadata": {
        "id": "qHPMNsCPHhIx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# To store data\n",
        "import pandas as pd\n",
        "\n",
        "# To do linear algebra\n",
        "import numpy as np\n",
        "\n",
        "# To create models\n",
        "from sklearn.decomposition import LatentDirichletAllocation\n",
        "from gensim.models.ldamulticore import LdaMulticore\n",
        "from gensim.models import LdaModel, CoherenceModel\n",
        "from gensim import corpora\n",
        "\n",
        "# To search directories\n",
        "import os\n",
        "\n",
        "# To use regex\n",
        "import re\n",
        "\n",
        "# To get punctuation\n",
        "import string\n",
        "\n",
        "# To parse html\n",
        "from bs4 import BeautifulSoup\n",
        "\n",
        "# To get progression bars\n",
        "from tqdm import tqdm\n",
        "\n",
        "# To measure time\n",
        "from time import time\n",
        "\n",
        "# To get simple counters\n",
        "from collections import Counter\n",
        "\n",
        "# To process natural language\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import RegexpTokenizer\n",
        "nltk.download('stopwords')\n",
        "\n",
        "# To use sparse matrices\n",
        "from scipy.sparse import csr_matrix\n",
        "\n",
        "# To create plots\n",
        "import matplotlib.pyplot as plt"
      ],
      "metadata": {
        "id": "yAs0ZXTeAwit",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ca710710-153d-4d9c-f94f-b5e37155ac72"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Load data\n",
        "comments = df.commentBody.dropna().values\n",
        "    \n",
        "print('Loaded Comments: {}'.format(len(comments)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FUxFppWvA2be",
        "outputId": "02dc94c0-9c42-44a1-9bd0-9ea43f49cecc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Loaded Comments: 231449\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Number of comments to use in the LDA\n",
        "n = 5000\n",
        "\n",
        "# To remove punctuation\n",
        "re_punctuation = re.compile('['+string.punctuation+']')\n",
        "\n",
        "# To tokenize the comments\n",
        "tokenizer = RegexpTokenizer('\\w+')\n",
        "\n",
        "# Get stopwords\n",
        "stop = stopwords.words('english')\n",
        "\n",
        "# Iterate over all comments\n",
        "preprocessed_comments = []\n",
        "for comment in tqdm(np.random.choice(comments, n)):\n",
        "    # Remove html\n",
        "    comment = BeautifulSoup(comment, 'lxml').get_text().lower()\n",
        "    \n",
        "    # Remove punctuation\n",
        "    comment = re_punctuation.sub(' ', comment)\n",
        "    \n",
        "    # Tokenize comments\n",
        "    comment = tokenizer.tokenize(comment)\n",
        "    \n",
        "    # Remove stopwords\n",
        "    comment = [word for word in comment if word not in stop]\n",
        "    preprocessed_comments.append(comment)\n",
        "     \n",
        "# Count overall word frequency\n",
        "wordFrequency = Counter()\n",
        "for comment in preprocessed_comments:\n",
        "    wordFrequency.update(comment)\n",
        "print('Unique Words In Comments: {}'.format(len(wordFrequency)))\n",
        "\n",
        "# Remove rare words\n",
        "minimumWordOccurrences = 5\n",
        "texts = [[word for word in comment if wordFrequency[word] > minimumWordOccurrences] for comment in preprocessed_comments]\n",
        "\n",
        "\n",
        "# Create word dictionary\n",
        "dictionary = corpora.Dictionary(texts)\n",
        "vocabulary = [dictionary[i] for i in dictionary.keys()]\n",
        "print('Documents/Comments: {}'.format(len(texts)))\n",
        "\n",
        "\n",
        "# Create corpus\n",
        "corpus = [dictionary.doc2bow(doc) for doc in texts]\n",
        "\n",
        "\n",
        "# Create sparse matrix\n",
        "def makesparse(mycorpus, ncolumns):\n",
        "    data, row, col = [], [], []\n",
        "    for cc, doc in enumerate(mycorpus):\n",
        "        for word in doc:\n",
        "            row.append(cc)\n",
        "            col.append(word[0])\n",
        "            data.append(word[1])\n",
        "    X = csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(cc+1, ncolumns))\n",
        "    return X\n",
        "\n",
        "\n",
        "# Create sparse matrix\n",
        "X = makesparse(corpus, len(dictionary))\n",
        "print('Train Shape:\\t{}'.format(X.shape))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WlrUgcMdA80H",
        "outputId": "cc3509aa-c93a-4225-f076-00a06ee58d1d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 5000/5000 [00:04<00:00, 1135.63it/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Unique Words In Comments: 21383\n",
            "Documents/Comments: 5000\n",
            "Train Shape:\t(5000, 5089)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Sklearn LDA Model"
      ],
      "metadata": {
        "id": "q0z11CZa3A5n"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "numberTopics = 20\n",
        "print('Number of topics:\\t{}'.format(numberTopics))\n",
        "#Model Creation\n",
        "model_sklearn = LatentDirichletAllocation(n_components=numberTopics, \n",
        "                                          learning_method='online',\n",
        "                                          n_jobs=16,\n",
        "                                          max_iter = 1,\n",
        "                                          total_samples = 10000,\n",
        "                                          batch_size = 20)\n",
        "\n",
        "perplexity_sklearn = []\n",
        "timestamps_sklearn = []\n",
        "start = time()\n",
        "for _ in tqdm(range(100)):\n",
        "    model_sklearn.partial_fit(X)\n",
        "    # Append the models metric\n",
        "    perplexity_sklearn.append(model_sklearn.perplexity(X))\n",
        "    timestamps_sklearn.append(time()-start)\n",
        "    \n",
        "# Plot the topics\n",
        "for i, topic in enumerate(model_sklearn.components_.argsort(axis=1)[:, -10:][:, ::-1], 1):\n",
        "    print('Topic {}: {}'.format(i, ' '.join([vocabulary[id] for id in topic])))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FmzU8Nd9B9Ap",
        "outputId": "fc2d8d1e-bb8e-4a41-8f26-6ac3d63c7f4e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Number of topics:\t20\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\r  0%|          | 0/100 [00:00<?, ?it/s]"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Gensim LDA Model"
      ],
      "metadata": {
        "id": "jIG3xqhl3C9o"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model_gensim = LdaModel(num_topics=numberTopics,\n",
        "                        id2word=dictionary,\n",
        "                        iterations=10,\n",
        "                        passes=1,\n",
        "                        chunksize=50,\n",
        "                        alpha='auto',\n",
        "                        eta='auto',\n",
        "                        update_every=1)\n",
        "\n",
        "\n",
        "perplexity_gensim = []\n",
        "timestamps_gensim = []\n",
        "start = time()\n",
        "for _ in tqdm(range(100)):\n",
        "    # Online update of the model\n",
        "    model_gensim.update(corpus)\n",
        "    # To compare sklearn and gensim the perplexity has to be transformed by np.exp(-1*x)\n",
        "    perplexity_gensim.append(np.exp(-1 * model_gensim.log_perplexity(corpus)))\n",
        "    timestamps_gensim.append(time() - start)\n",
        "    \n",
        "    \n",
        "    \n",
        "# Plot the topics\n",
        "for i, topic in enumerate(model_gensim.get_topics().argsort(axis=1)[:, -10:][:, ::-1], 1):\n",
        "    print('Topic {}: {}'.format(i, ' '.join([vocabulary[id] for id in topic])))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "K3Qkfz5yCKhg",
        "outputId": "abd6b63e-98af-42f8-fffc-06c7c3490b37"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "  5%|▌         | 5/100 [00:26<08:33,  5.40s/it]/usr/local/lib/python3.7/dist-packages/gensim/models/ldamodel.py:1023: RuntimeWarning: divide by zero encountered in log\n",
            "  diff = np.log(self.expElogbeta)\n",
            "100%|██████████| 100/100 [08:18<00:00,  4.99s/it]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Topic 1: power true bannon order else liberal looks fight week move\n",
            "Topic 2: world want americans day put change keep anything wall stand\n",
            "Topic 3: money actually maybe making working says mean try fake society\n",
            "Topic 4: may back around bad stop thank lies remember love given\n",
            "Topic 5: really better without voted hillary voters big plan either means\n",
            "Topic 6: state war putin administration children wrong run mind actions policies\n",
            "Topic 7: public donald instead history cannot story education reason future likely\n",
            "Topic 8: people one would us many get even country time think\n",
            "Topic 9: long news feel self kind issues agree especially quite problems\n",
            "Topic 10: trump president first states congress must hope office done united\n",
            "Topic 11: like obama man media white another house family place whether\n",
            "Topic 12: care health insurance countries system aca cost federal deal market\n",
            "Topic 13: political made ever help yes pay law act rather reality\n",
            "Topic 14: america nothing might next old men different simply etc getting\n",
            "Topic 15: life point least live social used makes security living expect\n",
            "Topic 16: could mr new times article russia facts russian intelligence though\n",
            "Topic 17: republicans republican election women party democrats gop vote clinton support\n",
            "Topic 18: years american every said last tax anyone high enough end\n",
            "Topic 19: never real thing best free press call problem 2 3\n",
            "Topic 20: right well going go work great left jobs show tell\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "**Sklearn LDA Topic Modeling**\n",
        "\n",
        "Topic 1: american last world political far party change every education economic\n",
        "\n",
        "Topic 2: power gop everyone past republicans based along political liberals wonder\n",
        "\n",
        "Topic 3: good need another take many make home self simply could\n",
        "\n",
        "Topic 4: americans states best right people united social could one american\n",
        "\n",
        "Topic 5: people must money clinton business russia take countries pay never\n",
        "\n",
        "Topic 6: us still tax would point part government live job nyt\n",
        "\n",
        "Topic 7: times news like service single military side guy hand top\n",
        "\n",
        "Topic 8: care health insurance 1 family 2 plan year much system\n",
        "\n",
        "Topic 9: obama also president administration republican time people democratic lost office\n",
        "\n",
        "Topic 10: nation story including healthcare speech poor com large general www\n",
        "\n",
        "Topic 11: women go back people time better may white us way\n",
        "\n",
        "Topic 12: trump put putin enough country government hope intelligence man information\n",
        "\n",
        "Topic 13: trump going get could bad mexico way president wall presidency\n",
        "\n",
        "Topic 14: like one would know see get even think mr democrats\n",
        "\n",
        "Topic 15: work jobs russian important evidence second game using 50 saw\n",
        "\n",
        "Topic 16: vote election policy lie democracy thank foreign votes citizens voting\n",
        "\n",
        "Topic 17: trump president right america people donald make got yes others\n",
        "\n",
        "Topic 18: press long please voters control stop continue call fire follow\n",
        "\n",
        "Topic 19: years trump new never state media true real every bannon\n",
        "\n",
        "Topic 20: public rights etc politics words school law line deal constitution\n",
        "\n",
        "\n",
        "**Gensim LDA Topic Modeling**\n",
        "\n",
        "Topic 1: take must power seems means fight taking civil interest military\n",
        "\n",
        "Topic 2: years fact state old become past reality act makes however\n",
        "\n",
        "Topic 3: say news another real congress enough bannon fear getting security\n",
        "\n",
        "Topic 4: even world could america also better first life nothing since\n",
        "\n",
        "Topic 5: obama nation clinton done job part voted wrong information perhaps\n",
        "\n",
        "Topic 6: office someone u show making policies laws came immigration current\n",
        "\n",
        "Topic 7: every day college anti reading second respect due difficult strong\n",
        "\n",
        "Topic 8: new money might support far matter reason rest votes york\n",
        "\n",
        "Topic 9: give march 000 liberal city living rich group went companies\n",
        "\n",
        "Topic 10: care health insurance voters thank high plan family given poor\n",
        "\n",
        "Topic 11: people one us get country see way know american well\n",
        "\n",
        "Topic 12: right election democrats political hillary problem history course lost democratic\n",
        "\n",
        "Topic 13: trump would president man donald putin russia administration sure campaign\n",
        "\n",
        "Topic 14: business russian actions love idea evidence china god brooks moral\n",
        "\n",
        "Topic 15: said states read use united policy constitution comments wants almost\n",
        "\n",
        "Topic 16: time many never go back white really great want times\n",
        "\n",
        "Topic 17: mr may got bad true lot feel important says turn\n",
        "\n",
        "Topic 18: much going government work media thing last house pay put\n",
        "\n",
        "Topic 19: women public tax truth law men rights children politics national\n",
        "\n",
        "Topic 20: like think make good need long two ever something year"
      ],
      "metadata": {
        "id": "Wko_dbdj3M_r"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Sentiment Analysis of the comments text (commentBody) (EDA)\n",
        "\n",
        "natural language toolkit - sentimentintensityanalyzer\n",
        "\n",
        "compound - The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive). \n",
        "\n",
        "vader_lexicon - VADER (Valence Aware Dictionary and sentiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media"
      ],
      "metadata": {
        "id": "bM4hW7uMHbos"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import nltk\n",
        "nltk.download('vader_lexicon')\n",
        "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
        "sid = SentimentIntensityAnalyzer()\n",
        "\n",
        "\n",
        "#make copy of combined dataset\n",
        "df1 = df\n",
        "\n",
        "#compound score\n",
        "df1['sentiment'] = df1['commentBody'].map(lambda x: sid.polarity_scores(x)['compound'])\n",
        "\n",
        "\n",
        "df1.head()\n"
      ],
      "metadata": {
        "id": "uucBeszEbMO0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 750
        },
        "outputId": "791eca01-b312-4e58-df46-3c4dc4d8043a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[nltk_data] Downloading package vader_lexicon to /root/nltk_data...\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                  articleID                author  \\\n",
              "0  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "1  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "2  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "3  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "4  58691a5795d0e039260788b9   jennifer steinhauer   \n",
              "\n",
              "                                            headline  \\\n",
              "0   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "1   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "2   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "3   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "4   G.O.P. Leadership Poised to Topple Obama’s Pi...   \n",
              "\n",
              "                                            keywords newDesk_x  \\\n",
              "0  ['United States Politics and Government', 'Law...  National   \n",
              "1  ['United States Politics and Government', 'Law...  National   \n",
              "2  ['United States Politics and Government', 'Law...  National   \n",
              "3  ['United States Politics and Government', 'Law...  National   \n",
              "4  ['United States Politics and Government', 'Law...  National   \n",
              "\n",
              "               pubDate sectionName_x  \\\n",
              "0  2017-01-01 15:03:38      Politics   \n",
              "1  2017-01-01 15:03:38      Politics   \n",
              "2  2017-01-01 15:03:38      Politics   \n",
              "3  2017-01-01 15:03:38      Politics   \n",
              "4  2017-01-01 15:03:38      Politics   \n",
              "\n",
              "                                             snippet typeOfMaterial_x  \\\n",
              "0  The most powerful and ambitious Republican-led...             News   \n",
              "1  The most powerful and ambitious Republican-led...             News   \n",
              "2  The most powerful and ambitious Republican-led...             News   \n",
              "3  The most powerful and ambitious Republican-led...             News   \n",
              "4  The most powerful and ambitious Republican-led...             News   \n",
              "\n",
              "                                              webURL  ...  recommendedFlag  \\\n",
              "0  https://www.nytimes.com/2017/01/01/us/politics...  ...              NaN   \n",
              "1  https://www.nytimes.com/2017/01/01/us/politics...  ...              NaN   \n",
              "2  https://www.nytimes.com/2017/01/01/us/politics...  ...              NaN   \n",
              "3  https://www.nytimes.com/2017/01/01/us/politics...  ...              NaN   \n",
              "4  https://www.nytimes.com/2017/01/01/us/politics...  ...              NaN   \n",
              "\n",
              "   replyCount  sectionName_y sharing    status  userDisplayName      userID  \\\n",
              "0           0       Politics       0  approved         N. Smith  64679318.0   \n",
              "1           0       Politics       0  approved      Kilocharlie  69254188.0   \n",
              "2           0       Politics       0  approved      Frank Fryer  76788711.0   \n",
              "3           0       Politics       0  approved      James Young  72718862.0   \n",
              "4           0       Politics       0  approved               M.   7529267.0   \n",
              "\n",
              "    userLocation  typeOfMaterial_y  sentiment  \n",
              "0  New York City              News     0.9638  \n",
              "1        Phoenix              News    -0.7906  \n",
              "2        Florida              News     0.8360  \n",
              "3        Seattle              News    -0.7478  \n",
              "4        Seattle              News     0.0000  \n",
              "\n",
              "[5 rows x 37 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-50d81116-c808-41ec-b54d-793c19c3d345\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>articleID</th>\n",
              "      <th>author</th>\n",
              "      <th>headline</th>\n",
              "      <th>keywords</th>\n",
              "      <th>newDesk_x</th>\n",
              "      <th>pubDate</th>\n",
              "      <th>sectionName_x</th>\n",
              "      <th>snippet</th>\n",
              "      <th>typeOfMaterial_x</th>\n",
              "      <th>webURL</th>\n",
              "      <th>...</th>\n",
              "      <th>recommendedFlag</th>\n",
              "      <th>replyCount</th>\n",
              "      <th>sectionName_y</th>\n",
              "      <th>sharing</th>\n",
              "      <th>status</th>\n",
              "      <th>userDisplayName</th>\n",
              "      <th>userID</th>\n",
              "      <th>userLocation</th>\n",
              "      <th>typeOfMaterial_y</th>\n",
              "      <th>sentiment</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>N. Smith</td>\n",
              "      <td>64679318.0</td>\n",
              "      <td>New York City</td>\n",
              "      <td>News</td>\n",
              "      <td>0.9638</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>Kilocharlie</td>\n",
              "      <td>69254188.0</td>\n",
              "      <td>Phoenix</td>\n",
              "      <td>News</td>\n",
              "      <td>-0.7906</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>Frank Fryer</td>\n",
              "      <td>76788711.0</td>\n",
              "      <td>Florida</td>\n",
              "      <td>News</td>\n",
              "      <td>0.8360</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>James Young</td>\n",
              "      <td>72718862.0</td>\n",
              "      <td>Seattle</td>\n",
              "      <td>News</td>\n",
              "      <td>-0.7478</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>58691a5795d0e039260788b9</td>\n",
              "      <td>jennifer steinhauer</td>\n",
              "      <td>G.O.P. Leadership Poised to Topple Obama’s Pi...</td>\n",
              "      <td>['United States Politics and Government', 'Law...</td>\n",
              "      <td>National</td>\n",
              "      <td>2017-01-01 15:03:38</td>\n",
              "      <td>Politics</td>\n",
              "      <td>The most powerful and ambitious Republican-led...</td>\n",
              "      <td>News</td>\n",
              "      <td>https://www.nytimes.com/2017/01/01/us/politics...</td>\n",
              "      <td>...</td>\n",
              "      <td>NaN</td>\n",
              "      <td>0</td>\n",
              "      <td>Politics</td>\n",
              "      <td>0</td>\n",
              "      <td>approved</td>\n",
              "      <td>M.</td>\n",
              "      <td>7529267.0</td>\n",
              "      <td>Seattle</td>\n",
              "      <td>News</td>\n",
              "      <td>0.0000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 37 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-50d81116-c808-41ec-b54d-793c19c3d345')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-50d81116-c808-41ec-b54d-793c19c3d345 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-50d81116-c808-41ec-b54d-793c19c3d345');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 100
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import seaborn as sns\n",
        "mean_sent = df1[['sectionName_y','sentiment']].groupby('sectionName_y').mean()\n",
        "\n",
        "mean_sent['year'] = 2017\n",
        "mean_sent['section'] = mean_sent.index\n",
        "\n",
        "sent_mat = mean_sent.pivot('section', 'year', 'sentiment')\n",
        "\n",
        "plt.figure(figsize=(10,10))\n",
        "sns.set(font_scale=1.75)\n",
        "plt.title('Mean Sentiment of Comments by Section for Jan 2017')\n",
        "g = sns.heatmap(sent_mat, square=True, cmap=\"Blues\")\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "rDATpU7dbMLD",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 649
        },
        "outputId": "ea658ecb-c744-4b3c-8d13-1facd8f6fb60"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x720 with 2 Axes>"
            ],
            "image/png": "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8x80aBDjx4+3WPrx9PTEwcHBrO3o0aPcvHmTv/zlLzg5OZldMxgM+Pj4cOjQIRITE/OdIK1ateqOm7TT09M5efIkjRs3pmbNmhbXfX192bFjB4mJifj4+Jj2JOW11OHu7k79+vU5fPgw6enpBXopaJs2bSzaWrZsib29PUlJSWbtBw4cYOHChezdu5cLFy6QmZlpupb75ymW7Epx9aRdu3YsX76czz77jKioKNPLal9++WX69euXrxjOzs4sXbqU0NBQNm7cSExMDDVq1CAoKIiJEyfqPWxiM5QgSanSpUsX5s+fn6++NWrUsGjLefomr2t/br/TUzoFVdDxcv6sXr36HfsfPnyYq1evFihByiuenZ0dVatWNXuqKS4ujqCgIOzs7OjQoQMeHh5UqFABg8FAZGTkPc/JkdKvefPmLFy48J797rbM7OrqyrRp05g2bZo1pybyQFGCJA+svPZB5CQVqamped5z/vx5s373q6Dj5fyZ036n/hUrVizQPPKKl52dzcWLF82W1L744gsyMzP55ptvaN26tVn/9evXF2hMW1V660ciYk16zF/KlEcffRQnJyfi4+MtHnEHiI2NBW7vIbIGFxcX6tatS3Jycp5JUs67r3I2zOb8mTOPPzt79izHjh2jXr16pkSqXLlyANy6deuu8/j5558t2vbt20dWVhaenp6mtuPHj1OlShWL5Cg1NbXIjkAQEXkQKUGSMsXR0ZHevXuTmprKokWLzK5FRkaSlJSEr69vvvcf5Uf//v3JyMjg008/NWuPjo5mx44dPPzww6aEpE2bNtSrV49t27YRFxdn1j8kJITMzEz69+9vaqtUqRIGg4HTp0/fdQ5r1qzh999/N/1zVlaW8xsG3wAAIABJREFUaT5/jle7dm0uXbrEkSNHTG0ZGRnMnDnTbC+S3IWhGD8iUmK0xCZlzuuvv05MTAwhISHExsbSrFkzjh07xubNm6lSpQozZsyw6njjxo1j+/bthIeHc/jwYXx8fDh9+jRRUVE4OTnx3nvvYWd3+3cROzs73nvvPcaOHcszzzxjOgcpJiaGffv20axZM7OnhCpWrIi3tzexsbG8/vrreHh4YGdnx5NPPml2FlL79u0ZOnQoffr0oVKlSuzcuZPDhw/TrVs30xNsACNHjuT7779nxIgR9OrVC3t7e3744QeysrJo0qTJHQ+2FBGxNUqQpMypXr06K1euZM6cOWzbto3o6GgqV65Mv379CA4OtvpBiOXLlycsLIwvvviCqKgoFi9eTMWKFenSpQsTJ060WM7z9fVlxYoVzJkzh507d3L16lUeeughnn/+eZ5//nnKly9v1v+f//wn77//Pjt27ODKlSsYjUbatGljliCNGTOGLl26sHTpUo4fP06NGjV44YUXePHFF81ide/enZCQEL744gu+/fZbXFxc6NixI6+99prFMQWSt+J8Wa2IlByD0Wg0lvQkRKTsm7HpsHXi9GhklTiFFX3kUrGN5degcrGNJSLmVEESESmAUnwMkohYkTZpi4iIiOSiCpKIFIup3Up2acxaVEASsQ2qIImIiIjkogRJREREJBctsYmIFITW2ERsgipIIiIiIrmogiQiUgA6KFLENqiCJCIiIpKLKkgiIgWggyJFbIMqSCIiIiK5qIIkIlIAKiCJ2AZVkERERERyUQVJRKQgVEISsQmqIImIiIjkogqSiEgB6BwkEdugCpKIiIhILqogiYgUgM5BErENqiCJiIiI5KIKkohIAaiAJGIblCCJSLHwnLzWKnFSPutrlTgiInejJTYRERGRXFRBErGSq1evsnjxYjZv3kxKSgrZ2dlUq1aNhx9+GB8fH4YPH07NmjVLeppyv7TGJmITlCCJWEF6ejojRozg0KFD1K9fn379+lG1alVSU1P55ZdfCA0NpUWLFkqQREQeEEqQRKzgq6++4tChQwwbNoy///3vGHI9C3748GEqVqxYQrMTa9JBkSK2QQmSiBXs378fgBEjRlgkRwCNGjWyaNu8eTOLFy/m4MGDGI1GHn30UUaMGMGQIUMs+mZnZ7Nq1SoiIiI4dOgQ2dnZPPTQQ3Ts2JEXX3yRypUrm/qmpKQwf/58fvjhB1JTU6lSpQqenp4EBgbSvXt3i5grV67k999/B8DT05MxY8bQo0cPs/GnTp1KZGQkmzdvZtOmTaxevZo//viDESNG8NZbbxXuhyYiUoopQRKxgpwE5dixYzRt2vSe/RcsWMDHH39MtWrVGDBgAA4ODmzatInp06fz22+/8c4775j6ZmdnExwczJYtW6hTpw79+/fHycmJ48ePEx4ezoABA0zjx8bG8txzz3Hjxg06d+5Mo0aNuHjxIvv372f16tWmBMloNDJ58mSioqJo2LAh/fv3B2Dnzp0EBwfz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efvtt+nUqRPly5fH39+fmTNn8t133xEUFGQWZ8OGDWRmZtK3b9+7jrdmzRoyMzMJCwujSZMmZtcuXrx4z+/80UcfcfLkSYKDg832aq1atYq33nqLGTNmsHTpUrN7zp49i7e3N2FhYTg6OgK3k72goCAWL15Mhw4d7jmuiMiDSBUksTkXLlwgKiqKTp06mSVHAB4eHgwdOpRDhw5x6NChO8bIzs7m3//+Nw0bNjRLjgCqVavGmDFjuHjxIj/++CNwe0N4165dSUhI4OjRo2b9165di4ODA7169crX/J2cnCza7rWZOyMjg/Xr11O9enXGjRtndm3QoEE0btyYmJgYTp06ZXHvtGnTTMkRQPv27alTpw4HDhzI13zLHJWQRGyCKkhic+Lj48nOzub69et5njOUs9R09OhRGjdunGeMY8eOcfnyZR566KE8YyQnJ5tidO3aFYB+/fqxceNG1q5dy8svvwzAqVOn+Pnnn+nSpQvVqlW767z9/f0JCwtj2LBh9OnTh/bt29O2bdt73pczj5s3b/KXv/zFIsEyGAz4+Phw6NAhEhMTzTZ4V6pUKc9lP3d3d/bt23fPcUVEHlRKkMTmXLp0CYDY2FhiY2Pv2O9uj8mnpaUBkJSURFJSUr5idOrUicqVK5slSOvWrcNoNNKvX797zrtVq1YsWbKE+fPns3LlSpYvX25Kbt544w28vb3veG96ejoANWrUyPN6TntOvxx3OgrB3t6e7Ozse865LNI5SCK2QQmS2BwXFxcAxo0bx2uvvXZfMXr37m164/m9ODo64u/vz4oVK/jll19o1aoVa9eupWLFinTr1i1fMXx9ffH19eX69evs27ePLVu2sGLFCsaOHUtUVNQdq0k5801NTc3z+vnz5836iYjYOu1BEpvj7e2NwWBg//79hY7RoEEDKlasyIEDB7h161a+78upFH333XckJiZy6NAhevToQfny5Qs0foUKFWjfvj1/+9vfGDFiBGlpaezdu/eO/R999FGcnJyIj48nIyPD4npOJS335m+xZDAU30dESo4SJLE5NWvWpGfPnsTExFg8tQVgNBotzjHKzd7enuHDh3P8+HE++eSTPJOk/fv3WyzTtWnThjp16hAVFWV6ND8/y2sAv/zyS57JTU71J6/N2zkcHR3p3bs3qampLFq0yOxaZGQkSUlJ+Pr66oBJEZH/T0ts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R44c4eDBg/c5cxGRO1OCJCL3xdHRkYEDB3Lx4kXWrFmDq6srvXv3Nuvj7u7OoEGDOHnyJK+++irnzp2ziJOSksLGjRtN//zII48AsGbNGrN9RhcvXuTNN9/M82BJERFr0RKbiNy34cOHs2jRIoxGI3379qVChQoWfd566y1OnjzJxo0b2bVrF02bNqV27dpcv36d33//nZSUFLp3707Pnj0B6NatG40bNyYhIYG//vWvtGnThqysLKKjo6lZsyZPPPEEW7ZsKe6vKiI2QhUkEblvDz/8MLVr1wb+d/ZRbuXLl2fBggV8+OGHtGjRgqNHj7Jx40bi4+OpVq0awcHBvP7666b+Dg4OLFu2jFGjRuHs7MyOHTtISkpiwIABhIeHm52lJCJibXrViIjct19//ZUhQ4bQokUL01NsIiIPMlWQROS+ff755wCMHDmyhGciImId2oMkIoWyd+9eVq1axZEjR9i3bx+NGzcmICCgpKclImIVSpBEpFCSk5NZvXo1FStWpHPnzvztb3+jXLlyJT0tERGr0B4kERERkVy0B0lEREQkFyVIIiIiIrkoQRIRERHJRQmSiIiISC5KkERERERyUYIkIiIiksv/A202gsGgVaX+AAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "Comments 2017.\n",
        "The more Compound score closer to +1, the higher the positivity of the text.\n",
        "\n",
        "Frpm the above plots we can infer that the sentiment of the readers is overall positive based on the mean compound scores for topics on\n",
        "Africa, Americas, Asia Pacific Baseball etc. \n",
        "\n",
        "On the other hand sentiments are negative towards trends topics like\n",
        "Sunday Review, Television, Tennis, The Daily\n",
        "\n",
        "The above provides NYT with an overview of the sentiments towards different article topics and helps NYT get a high level overview of those topics that were received positively/negatively for the articles in the year 2017.\n",
        "\n",
        "We also not that the topics/section: politics, pro basketball, pro football, room for debate have a mean zero score, indicating these comments have a neutral sentiment."
      ],
      "metadata": {
        "id": "axVbaftwjdRi"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "In particular, the New York\n",
        "Times has a feature called “NYT Picks” which are a professionally curated set of “the\n",
        "most interesting and thoughtful” comments.\n",
        "(1) These comments are made available in a\n",
        "filtered tab within the interface that sets them apart and labels them as “NYT Picks”. The\n",
        "New York Times pre-moderates all comments on the site, meaning that no comment is\n",
        "published without it first being read by a moderator. This process ensures a generally\n",
        "high quality level for comments since the negative criteria for comment exclusion such as\n",
        "obscenity, personal attacks, or other spam have already been applied"
      ],
      "metadata": {
        "id": "x10ZmgVtIv9k"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#check which comments have been picked up by EditorsSelection\n",
        "n_bins=20\n",
        "plt.figure(figsize=(5,5))\n",
        "plt.hist(df1.loc[df1.editorsSelection==0, 'sentiment'], bins = n_bins)\n",
        "plt.xlabel('mean sentiment score')\n",
        "plt.ylabel('count of comments')\n",
        "plt.title('Polarity score of comments that are not editors selection')\n",
        "\n",
        "plt.figure(figsize=(5,5))\n",
        "plt.hist(df1.loc[df1.editorsSelection==1, 'sentiment'], bins = n_bins)\n",
        "plt.xlabel('mean sentiment score')\n",
        "plt.ylabel('count of comments')\n",
        "plt.title('Polarity score of comments that are Editors selection')\n",
        "\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "HjtbiGo0i-xC",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 737
        },
        "outputId": "cd8756b9-c1c7-4b0d-f89b-a8e3bf24f6a7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 360x360 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmIAAAFoCAYAAAD0JZcdAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeVxO6f8/8NfdplIoFUqK4a4mS0pli7EzxhCyJ0v25WMby8xgZsyMfTCYIYx9FiFZsqVRQpuQEkaSFBUVJbSd3x9+9/3tdt/VfSdu8Xo+HvMYnXOdc97nnPs+531f13WuIxIEQQARERERvXMa6g6AiIiI6GPFRIyIiIhITZiIEREREakJEzEiIiIiNWEiRkRERKQmTMSIiIiI1ESpRMzGxkbuPycnJ3h4eGDHjh0oKCh4oyAOHjwIGxsbrF+//o3WoywbGxt07tz5nWyLKkdhYSHWrFmDLl26wN7eHjY2Nvjpp5/UHRa9hzp37gwbGxt1h0FVkKJ7Q3h4OGxsbDB//nw1RVX1vM/HzNPTEzY2Nrh//766Q5FSqUasR48ecHd3R79+/WBnZ4f4+HgsXboUY8eORX5+/tuK8Z24f/8+bGxs4Onpqe5QSIGdO3di06ZNKCgoQPfu3eHu7o7mzZurOywq4V0lQO/LD6n58+fDxsYG4eHh6g6lynqfb9jledcVCKSc9+X6oAotVQrPnTsX9evXl/4dHx8PT09PhIeHw9fXF8OHD6/0AN+GgIAAaGtrqzsMUsGZM2cAAHv37oWlpaWaoyGij0Xz5s0REBAAQ0NDdYdClWD58uV4/vw56tSpo+5QpN6oj5idnR1GjRoFAAgMDKyMeN6JTz75BA0aNFB3GKSCtLQ0AGASRkTvlJ6eHj755BOYmZmpOxSqBObm5vjkk0/eq8oYlWrEFPn0008BAKmpqTLTg4ODsXPnTsTGxuL58+cwNzdHly5dMGHCBNSsWVOpdaelpeHw4cMICQnBvXv38PjxYxgYGKBFixYYPXo0WrduLbeMp6cnIiIicObMGVy+fBm7d+/Gf//9B01NTURFRQF4VXVpYWGBoKAgAMD69euxYcMGAEBERIRM84q7uztmzZqFTp06wcjICGfPnoWWlvxhCwwMxJQpU9C5c2f8/vvv5e5bamoqtm7divPnzyMtLQ1aWlowMTGBg4MDhg8fjmbNmsmUz8rKwo4dOxAUFITk5GRoaGjAwsICbm5uGDVqlMxForCwEH///TcOHjyIxMREAECjRo3Qv39/DB48WC5+ZY6ZIAg4duwY9u3bh/j4eLx48QL169fH559/Dm9vb+jp6ZW7zyVduXIFPj4+iI6ORm5uLszMzODm5oZJkyahbt26crFJlDw3N2/eVGpbV69exY4dOxAVFYWsrCzUqlULn3zyCXr37o1BgwbJlH3w4AF+//13nDt3DhkZGTA0NISjoyPGjx+PFi1ayJS9f/8+unTpAhcXF/j4+GDDhg04fvw4Hj16hAYNGmDcuHHo27cvgFdNML/99hvi4uIgCALatWuHr7/+WmZfgVfNXX5+fti1axcAYOPGjYiNjYW2tjbat2+PefPmwczMDM+fP8fGjRsREBCAjIwMmJubY+zYsXL7U3K/tmzZgpCQEKSlpUFPTw8tW7bEhAkT4OjoKFM2PDwcI0eOhLu7OxYsWIC1a9ciMDAQWVlZsLCwwODBgzF69GiIRCKZ8orOUcnvWV5eHvbu3Ytjx44hJSUFBQUFMDY2hlgsxpdffonPP/+8zPN48OBBLFiwAACQkpIisx0XFxfs3r1bbpn9+/dj9+7dSExMhJ6eHjp06IA5c+bI/RpW9VpTctsl9x14VXtbsuVAkevXr+PYsWO4ePEiHjx4gJycHJiamqJ169aYOHEirKys5JaRXLdOnDgBHx8fHD16FPfv30eHDh3w22+/AQCeP3+OXbt24fjx40hKSgIANGnSBEOHDoW7u3uZMZUkOdZTp07FwIED8csvvyA0NBS5ubmwtrbG2LFj0a9fP4XLKvvdlnzWAcDPz0/6bwCYOnUqpk2bplSsISEh2LNnD2JiYpCbm4s6deqga9eumDhxIoyMjOTKZ2dn49dff8WpU6fw5MkTWFpaYvDgwXLnUaLk92HZsmUAZK9LGzZskN4/AGDp0qXo37+/9G9V7oWSe9HSpUvRuHFjbNy4EVeuXEF2djYOHToEOzs7ZGZmYufOnQgMDERqaipEIhFq164Ne3t7DBo0CG3btlXquF25cgXbtm1DXFwcMjIyYGBggLp168LZ2Rnjx4+HiYmJTPmEhARs2bIFFy9exOPHj1GjRg24urpi8uTJaNKkiVLblFD1nEnuPwcOHMD169eRl5cHU1NTNGvWDEOGDEGbNm2Uvj6UvN+9/j1NSEjA77//jrCwMGRnZ6NWrVpo3bo1Jk2ahE8++USmrKrXybK8cSL27NkzAICOjo502ubNm/HLL79AU1MTzs7OMDY2RnR0NLZt24bAwEDs2bNHqV8Xp0+fxqpVq2BtbY3GjRvD0dERKSkpCA4ORnBwMJYtW1bqxcDHxwe+vr5wdHREp06d8ODBg1K3Y2dnhx49euDkyZMwMTGBm5ubdJ6TkxPMzMzQpUsXnDx5EmfPnkXXrl3l1rFv3z4AwODBg8vdr4cPH8Ld3R3Z2dmwsrKSbi81NRWHDx9GgwYNZBKxhIQEjBkzBg8fPoSJiQnat28PALh79y62bdsGR0dHaUxFRUWYPHkygoODUb16dbRt2xaCICAsLAw//PADzp07h99++w0aGvKVoaUds+LiYnz11Vc4evQo9PX10bRpU9SsWROxsbHYsGEDQkJCsHv3bujq6pa77wDg7++PBQsWoKioCI6OjqhXrx7i4uLw999/4/Tp09i1axcaN24MAHBzc4OFhQVOnjyJvLw8lW4mwKu+ZcuWLUNxcTHs7e3h7OyMrKws3Lx5EytWrJBJXG7evAkvLy9kZWWhYcOG6N69O1JTUxEYGIh///0XK1euRO/eveW2UVBQgNGjRyMxMRGurq7IyclBZGQk5s6dC0EQoKuri9mzZ6Np06Zo3749YmNjcfLkSdy8eROHDx9GtWrV5NZ55swZ7NmzB82bN4ebmxvi4uJw9OhRxMfHY9++fRg9ejTu3bsHFxcX5ObmIjIyEgsXLoSmpiYGDBggs67Lly9jwoQJePLkCRo2bIjPPvsMmZmZCA0Nxblz57Bq1SqFSdDTp08xePBg5OTkwMnJCTk5OYiKisLy5cvx7Nkz6Y3SxMQE7u7uCs+R5KJaVFSEMWPG4PLly6hVqxacnJygp6eHtLQ0RERE4MWLF+UmYg0aNIC7uzv8/Pygr6+PHj16SOc1atRIrvzKlSuxc+dOtGrVClZWVrhy5QoOHz6Ma9euwd/fX+a4q3qtcXd3x6VLl3Dv3j20b98epqam0nn6+vpl7gcAbNq0CYGBgbCxsYGDgwO0tLTw33//4eDBgzh9+jT+/PNPiMViueWKi4sxZcoUREVFwdnZGTY2NqhVqxYA4PHjxxg9ejRu3rwJU1NTODs7QxAEXL58GfPnz0dsbCwWLlxYbmwlpaamYsCAAdDV1UXr1q3x6NEjREVFYd68eSgqKpL7rKny3XZyckJGRgZCQ0PRoEEDODk5SddjZ2enVHyrVq3Cli1boK2tjWbNmsHU1BQ3b96U/mj966+/ZBKKJ0+eYNiwYUhISICpqSm6dOmCp0+fYsWKFbh3757Sx8XNzQ2FhYWIjo6Gra2tTLwlW1sqei+MjIzEokWLYG1tjXbt2iE9PR0ikQjPnj3D4MGDce/ePdSpUwdt27aFlpYWHj58iKCgIOk1vzz//vsvJk+ejOLiYjRr1gwtWrTAs2fPkJycjJ07d6J79+4yxy0wMBAzZ85Efn4+bG1t0aJFCzx8+BDHjx/Hv//+iy1btsDZ2VmpY6fqOSsqKsKsWbNw4sQJaGtrw9HRESYmJnjw4AGCg4NRUFCANm3aqHx9eN3FixcxadIkPH/+HJ9++ilcXFxw584dHDlyBGfOnMGmTZvg6uoqt5yy18kyCUoQi8WCWCwWkpOT5ebNnDlTEIvFwuzZswVBEISrV68Ktra2goODg3D58mVpuZcvXwrTp08XxGKxMGnSJJl1HDhwQBCLxcKvv/4qMz0+Pl64efOm3DZjYmIEJycnwcnJSXj27JnMvBEjRghisVho1qyZEB4eXur+dOrUSWZacnKyIBaLhREjRihc5vz584JYLBbGjx8vNy81NVWwtbUVOnbsKBQVFSlcvqRff/1VEIvFwvfffy83LyMjQ7h165b074KCAqFHjx6CWCwWlixZIrx8+VKm/K1bt4SkpCTp39u2bRPEYrHQu3dvISMjQzo9LS1Nup7t27fLrKO8Y7ZlyxZBLBYLw4cPF9LS0qTTX758KXz99deCWCwWVq5cWe5+C8KrY9W8eXPBzs5OOH36tHR6UVGR8NNPPwlisVjo27evUFxcLLNcp06dBLFYrNQ2JCIiIgQbGxuhZcuWwrlz52TmFRQUCGfPnpX+XVxcLHzxxReCWCwWVqxYIbP9EydOSD/TDx8+lE6XfGYkn5ucnBzpvNDQUEEsFgvt27cXXFxcZPb15cuX0mN+4MABmbjmzZsniMViwdbWVm4ZT09P6bl9fXshISEKP9c5OTlCu3btBFtbW8HPz09mXkxMjODs7Cw4ODgIjx8/lk4PCwuT7teUKVOEFy9eSOdduXJFsLOzExwcHITc3FyZ9ZV1jiTr7N+/v/D8+XOZec+fP5e5VpRH0X4qiqN169Yy14/c3FzBw8ND4XGvyLVGcq7CwsKUjl3iwoULQnp6utz0ffv2CWKxWPDy8pKbJzkn3bp1k/kcSowbN04Qi8XCDz/8IHPOMjIyhP79+wtisVgIDg5WKj7JNVmyvsLCQum848ePKzwHFfluSz4X8+bNUyqukgICAgSxWCx8/vnnQmJionR6cXGxsG7dOkEsFgszZsyQWWbRokWCWCwWxo4dK+Tl5UmnX716VXBwcFC4X6XFWNp9q+Q6Vb0XSu4NYrFY8PHxkVunZJsTJkyQOSeCIAhPnjwRYmNjFcbyOsn1JyAgQG7erVu3ZO4dycnJgoODg+Dg4CB3HQ0ODhbs7e2Fjh07ytybSjtmFTlnGzduFMRisdCrVy/h3r17MvOePn0qd88q7/og2feS+cyzZ8+Etm3bCmKxWNizZ49M+e3btwtisVho166dzGemotdJRSrUR0wQBKSkpGDVqlU4duwYRCIRhgwZAuBVZ+ri4mJ4enrCwcFBuoyOjg4WL14MPT09BAUFKfXoqK2trcJfhc2aNcPw4cORk5ODsLAwhcsOHDgQLi4uFdk9hdq0aQNra2ucO3cODx8+lJm3f/9+FBcXY+DAgQprml6XmZkJAAp/uZiYmMhU8546dQqJiYlo0aIFvvnmG5maR+BVs0PJX2CS6tf58+fL/KowMzOTPpkkafZ6naJjVlhYiK1bt0JPTw+//PKLzK83HR0dLFy4EKampti3bx+Ki4vL3XdfX1+8ePECvXr1kqlZ1NDQkDYZxcfHIzIystx1lcfHxweCIGD27NnSWkQJLS0tdOzYUfp3eHg4bt26BXNzc8yYMUOmOrlHjx7o1q0b8vLy4OvrK7cdDQ0NfP/99zAwMJBOa9euHezs7JCeng43NzeZfdXR0YGXlxcAlLqfvXv3lltG0nSSkJAgtz03NzfY2dkhJSUFKSkp0un79+9HRkYGRo4cKVd73KxZM0yePBl5eXk4fPiwXAzVq1fHDz/8IFNz1KJFC7i5uSEvLw9xcXEKY1dE8pl3cnKSqznV1dWVuVZUlunTp8tcP6pXr44xY8YAkD/ub3KtqYg2bdrI1KJJeHh4oGXLlggLC0Nubq7CZWfNmiXXtBofH4/g4GDY29vjm2++kTlnJiYmWLJkCQDgr7/+UilOCwsLzJs3D5qamtJpPXv2RJMmTeQ+a+/yuw28qlUEgNWrV8Pa2lo6XSQSYdq0abCzs8PJkyeln728vDwcOnQIGhoaWLRokUx3iubNm1f6w2Zvci8Ui8Xw9vaWmy7ZlzZt2sicEwCoUaMG7O3tlYpNsp7Xr4vAq3tKyXvHzp07kZeXh5kzZ8qV79ChA4YMGSKtnSqPqucsPz8f27dvh0gkwrp16+T6CBsaGlbKfV7SpaRly5Zyn4NRo0ahadOmyMjIwIkTJ+SWrYzrpEqJWJcuXWBjYwNbW1t07txZWr24aNEitGrVCgCkfYr69Okjt7yxsTHc3NwgCAIuXbqk1Dbz8/MRFBSEtWvXYtGiRZg/fz7mz58vfWRc0g/idZX9+KpIJMLgwYNRVFSE/fv3S6cXFxfjwIED0NTUxMCBA5Val+TLsmbNGgQHB5c59MfFixcBAH379i23rTk1NRWpqakwNjZW+AX77LPPUKtWLaSkpMglk4DiY3b9+nVkZWWhZcuWCqvQdXV1YW9vjydPnuDu3btlxgeU/fnQ0dGRNk9JylVUYWEhIiIioKmpKe2npUxcPXv2VNiJU5LEKIrL3NxcYdW3JEFWdC4k89LT0xXGU9Yy5W2v5DrPnz8PAOjevbvC7Uiag2JiYuTm2dvbw9jYWG56w4YNy4xdETs7O2hoaODAgQPw9fXFkydPlF62ojp06CA3razYK3qtqagnT57g0KFDWL58Ob755hvp9h49egRBEBRuTyQSKfyehoaGAgC6deum8Mfgp59+Cn19fVy7dk2lGF1dXeV+/AGKj+O7+m4Dr5phb9y4ASsrK9ja2srNF4lEcHR0RFFRkfRGGBcXhxcvXqBp06YKH9b64osv3jiukt7kXtipUyeF13vJvWPr1q0ICAiQdg1SlWQ9c+fOxbVr1yAIQqllJdeQkk19JUnu/YquISVV5JzFxsbi6dOnaNq0qcr90FRR1rkCyr7+V8Z1UqU+Yj169IC+vj5EIhH09fXRqFEjdO3aVebXmWSjFhYWCtch6RwneQquLP/99x8mTZqE5OTkUsuU9kGsV69euetXlbu7O9auXYuDBw9i8uTJ0NDQQEhICB48eIBOnTrJdbwuaz0XL17E0aNHMX78eOjo6MDe3h7t2rXDgAEDYG5uLi0r6aclOallkRz7ksu/zsLCAtnZ2UhLS5OLV9Exk/xau3DhQrljRGVlZSkdY2V8PsqSnZ2NFy9eoE6dOjI1R+XFVVon67LiKu28S/oKKXpMWjKvtCS8rGXK217JdUpqLIYNG6ZwGQlF566071D16tXltlMea2trzJ8/HytXrsS3336LhQsXonHjxnB1dUXfvn3fyphwiuIvLfY3udZUREBAAL799tsy16loXu3atRUmRpLzvHbtWqxdu7bUdao63mNpnzVFx/FdfbeB/9vfpKQkpa9L5V0fS4u7ot7keJT23WvTpg3Gjh2L7du3Y+bMmdDS0oKNjQ1at26N/v37S/vflWf27Nm4ffs2goKCEBQUhBo1asDBwQGfffYZ+vXrJz2/wP8da0U/bEoq7/pfkXMmqTAoWXv2NryNc6XKdfKNxhF7mwRBwP/+9z8kJydj0KBBGDp0KBo0aAB9fX1oaGjgn3/+waJFi0rN5BV1gH5TRkZG6NmzJ/z9/REaGooOHTpIO+mX9rSaIpqamli9ejXGjRuHoKAghIWF4erVq7h8+TJ8fHywbt06tQxIp+iYSY6vlZWV3NN1r5N0Gv7YlNccrUxztSrLqLI+SXOx5EdUaRTVsFUk7rJ4eXmhZ8+eCAwMxMWLFxEZGYk9e/Zgz549mDBhAmbNmlWp21M2/je91qgqNTUV8+bNgyAIWLBgATp27Ii6detCV1cXIpEIs2fPxtGjRxVur7TrmuQ8Ozk5VerQPJX9Gagskv01NTVVWHtcUlk/TN9XZd2/5s6di8GDB+PMmTO4ePEioqOjERcXh+3bt2PRokUYOnRoueuvU6cOfH19ER4ejuDgYERGRiI0NBQhISHYvHkz/vzzT+m9XnKsy3tQ6vWnyl/3oZ6zyviOvPFTk68zMzPD/fv3kZKSorAqUZIVlzeY2p07d5CQkAB7e3tp/4aSKruZQFlDhgyBv78/fH19YWNjg+DgYNStW1emv5GybG1tYWtri8mTJ+P58+fYuXMn1qxZg8WLF0sTMUm2fffu3XKfhpE0Hb4+lEhJyh5/CUm5Ro0aSR/dfhNmZmZITEx8489HeYyMjKCrq4tHjx7h2bNnMr/wSosLQKl9Fysrrnetbt26SExMxPjx49G0aVN1h4M6depg+PDhGD58OIqLi3H27FnMnj0bPj4+cHd3V6rmt7K962vN2bNnkZ+fjzFjxkjHYXzT7Ulqrrp27SrtB/euvavvNvB/+2tkZKT0dUnSJ6+062PJ/m6VobLuhYpYWVlhzJgxGDNmDAoKCuDv74+FCxfi559/Rp8+fZRqBdDU1ETbtm2l95WMjAwsWbIEJ0+exJo1a7B69WoAr471vXv3MG/ePIVDSyirIudMsszbvt9Lrv+lfQbe9vW/0n/uSNqLjxw5IjcvMzMT586dg0gkknlUWRFJHxJF1X75+fk4ffp0JUT7fyT9ggoLC8ss5+joCBsbG+kju4WFhRgwYIBcx0lV6enpYeLEiahZsybS09Ol+9+mTRsArx4LL+8Xubm5OczNzaVDE7wuODgY2dnZsLCwULoZtXnz5jA0NERERASys7NV3Ct5ZX0+8vPzERAQIFOuojQ1NeHi4oKioiKFHdFLi+vEiRMK353q7+9fKXG9a5KLbGV/XxRR9jskoaGhgc6dO6Ndu3YQBAG3b99WejvKbkMZFb3WSPa3qKhIpe09ffoUgOJmv4SEBMTHx6u0PuDVwyHAuznPpanId1vVz4xE3bp10ahRI9y+fVs6VmJ57O3toauri7i4OIVN0JL4lFVe7JV1L1QmjoEDB8LW1hb5+fkqDcNRkqmpKSZPngzgVVO9hOQa8qaDtlfknDVt2hQ1atTAtWvX3ur1oaxzBbz963+lJ2LDhw+HhoYGdu/ejatXr0qn5+fnY8mSJcjLy0Pnzp3LbeK0srKChoYGwsLCkJCQIJ1eUFCAn3/+ucIfttIYGRlBW1sbycnJ5V5YBw8ejIKCAuzevRsaGhrw8PBQaVuHDh1SOBjp5cuX8eTJExgaGkprcLp37w5ra2tcuXIFS5cuVdi3peRFZcSIEQBevcbh8ePH0ukZGRlYvnw5APkBKMuio6MDb29vPHv2DFOnTlV43NPS0nDo0CGl1jdw4EDo6uri+PHj0oE+gVfV1mvWrMHDhw9ha2ur9Jg0ZRk3bhxEIhFWrVolfehBorCwUOYpH1dXV4jFYqSmpmLdunUySe/p06dx8uRJ6OnpqXyu1W3IkCGoXbs2tm3bhr///lvus11YWIhz587h1q1bb7wtya9KRRfZsLAwhIaGym3/8ePHiI2NBVB6fyRF23n8+LE0oXlTFb3WlLW/ZZHU+vn7+8v0A8vKysLXX39doSSzRYsWaNeuHaKjo/H9998rfOLyxo0bCAkJUXndyqrId7uixxCAdBys6dOn4/r163Lzs7KypF1HgFd9dvr27YuioiIsWbIEL168kM6LjY3Fnj17VNp+ebFX1r2wpMDAQFy+fFlu+t27d3H37l1oaGgoNUbn9u3bFXYil3w+Sv4oGTNmDHR1dbF8+XKcPHlSbpn8/HycOHFC4QNgr1P1nEmeMBcEATNmzJBLoHNycmQG/AYqdn3o1asXTExMcPnyZezdu1dm3q5duxATEwNTU1P07NlT6XWqotKbJps3b44ZM2bgl19+wbBhw+Di4gIjIyNER0fjwYMHsLKywnfffVfuemrXrg0PDw/8888/6NevH1q3bo3q1atLRxkeMWKEyl+csujo6KB9+/b4999/0bdvX3z66afSweNeH7Swb9++WLVqFfLy8uDm5qbygwGnTp3CvHnzUL9+fdjY2EBPTw8PHz5EdHQ0AGDGjBnS0e+1tLSwfv16jBkzBjt37kRAQABatmwpfarq1q1b2Lhxo/Sx3lGjRknb/bt37y4dEfzixYt49uwZOnXqpPKLzcePH487d+7A398fn3/+Oezs7FC/fn0UFhYiMTERt2/fho2NTamD65Zkbm6OH374AQsWLMCkSZPg5OQkHfQxMTERxsbGWLVqlVKjEZfHxcUFX331FVauXIlRo0bB3t4eDRs2RFZWFm7cuIH8/HzpUzCShM3LywtbtmzBmTNn8OmnnyI1NRXR0dHQ1NTEjz/+WOWaJmvUqIHffvsNEydOxOLFi/H777+jSZMmqFmzJh49eoTr16/j6dOn2Lhxo8LhG1TRuXNnREREYNSoUXB1dYWenh6MjIwwZ84c3LhxA0uXLkWtWrXQtGlTGBkZ4enTp4iMjEReXh569Ogh9zaJsraze/duuLu7o2XLlqhWrRoaNmyo8HF/ZVT0WtOpUyds3LgRy5cvx/nz56XNNnPmzCmzCadz584Qi8WIi4tDt27d4OTkhMLCQoSHh8PU1BRdu3atUO3DypUr4e3tjT///BNHjx6Fra0tzMzMkJubi5s3b+LBgwcYOXJkuZ2uK6oi323JNTA2NhYDBw5EkyZNpDWlXbp0KXN7ffr0we3bt7Fp0yYMGDAAdnZ2sLS0hCAISE5Oxs2bN6Gvry/Tf3fWrFmIiIhAcHAwunbtCmdnZzx9+hTh4eEYNGiQ3E24LA4ODqhduzZOnjwJT09P1K9fHxoaGhgwYAAcHR0r7V5YUnh4OHbt2gVTU1PY29vD0NAQjx8/RmRkJAoKCjBmzBi5EfEV2bhxI1asWAGxWAxra2toaGjg9u3buHXrFvT09DBlyhRpWSsrK6xevRpz5szB9OnTYWVlhUaNGqF69ep4+PChdKT7Q4cOlftjqiLnbOLEiYiPj0dgYCB69eoFJycn6YCu169fR9u2bWWGsKjI9UFfXx+rVq3CxIkT8cMPP+DAgQNo2LAh7ty5g+vXr0NPTw+rV69W+Q0yyqr0RAwAJkyYAFtbW+zYsUPmtQ5jxozBhAkTlO7UvXjxYjRu3Bi+vr6IiIiAvr4+WrVqhenTp6v8GLYyfvrpJyxfvhwXLlzA0aNHUVRUpHD0aAMDAzRt2hQREREqddKXGD16NOrVq0w61FwAACAASURBVIfo6GhER0fj2bNn0lGevby85GqDxGIx/P39sXXrVgQFBSE4OBg6OjqoV68exo0bJ9NJUlNTE7/99hv++usv+Pn5SR89lrziaMiQISo3o2poaGDFihXo0aMH9u3bh2vXriE+Ph41atRA3bp1MXbs2HJHRS+pb9++aNCggfQ1KDExMTAxMcHgwYMxefJkpWtGlDF27Fi0aNECO3bsQHR0NG7duoVatWpBLBbLjZJvY2MDPz8//P777wgJCcHJkydhYGCALl26YPz48W9lrKt3wcHBAUeOHMGOHTsQHBwsTT4lo69369ZN2gT+Jjw9PfHkyRMcO3YMp06dQkFBASwsLDBnzhx06tQJ2dnZiIyMxK1bt6SvmrK3t8fAgQNLfWxckVmzZkEQBJw5cwbHjx9HYWEhXFxcKpyIARW71jRt2hQrV67E9u3bcf78eWkNy6RJk8pMxLS1tbF37178+uuvOHv2LM6ePQtTU1O4u7tj+vTpWLp0aYX2oXbt2vj777+xb98+HDt2DPHx8bh8+TJMTExgaWkJT09PhW+GqEwV+W6vX78eK1asQFRUFOLi4lBcXIy6deuWm4gBkI5ttWfPHun3u3r16qhTpw6GDh0qV4NRq1Yt/PXXX1i3bh0CAwMRGBiI+vXrY9asWRg9erRKiVi1atWwefNmrFmzBjExMYiMjIQgCHBycpI+2FRZ90KJ/v37Q0tLC5cuXUJsbCyePHkCExMTuLq6YtiwYUodMwD49ttvERoairi4OISGhkqP+dChQzFmzBi5Bz66du2Kw4cPY/v27bhw4QIuXLgALS0tmJmZoVOnTujWrZvcK4BKo+o5k1RG+Pv748CBA4iLi8PLly9hamqKzz77TO7hhIpeH9q0aYMDBw7g999/R3h4OG7evAkjIyN88cUXmDx5stL7VxEiobIeBfqIpKWloXPnzjA2NsbZs2ffuH8YERERfZzez2eT33OSTvpDhw5lEkZEREQVxhoxJd25cwfbtm1DSkoKLl68CFNTUxw/fhyGhobqDo2IiIiqqLfSR+xDlJGRgf3790NXVxetWrXCN998wySMiIiI3ghrxIiIiIjUhH3EiIiIiNSEiRgRERGRmrCPGL23srKeobhYuZbz2rUN8Pix/Gji9OHjuf94qXruNTREMDIq+72zRO8aEzF6bxUXC0onYpLy9HHiuf948dxTVcemSSIiIiI1YSJGREREpCZMxIiIiIjUhIkYERERkZowESMiIiJSEyZiRERERGrCRIyIiIhITZiIEREREakJEzEiIiIiNeHI+kT0XjGsoQfdaspfmvILit5iNEREbxcTMSJ6r+hW00Kf2f5Klz+yuu9bjIaI6O1i0yQRERGRmjARIyIiIlITJmJEREREasJEjIiIiEhNmIgRERERqQkTMSIiIiI1YSJGREREpCZMxIiIiIjUhIkYERERkZowESMiIiJSEyZiRERERGrCRIyIiIhITZiIEREREakJEzEiIiIiNWEiRkRERKQmTMSIiIiI1ISJGBEREZGaMBEjIiIiUhMmYkRERERqwkSMiIiISE2YiBERERGpCRMxIiIiIjVhIkZERESkJkzElHTv3j0cOXIE9+/fl5keGxuLYcOGwdnZGV9++SX+/fdfNUVIREREVQ0TMSX98ccfmDdvHjQ0/u+QZWVlYcyYMYiOjkZOTg5u3bqFadOm4caNG2qMlIiIiKoKJmJKunTpEsRiMczNzaXTDh48iKdPn2LEiBGIiorC6tWrUVhYiB07dqgvUCIiIqoymIgpKT09HRYWFjLTQkJCoK2tjRkzZsDAwAC9e/dG8+bNcfnyZTVFSURERFUJEzElvXjxAjo6OtK/i4qKEBMTg2bNmsHAwEA63dLSEunp6eoIkYiIiKoYJmJKqlOnDhISEqR/X758Gc+fP4eLi4tMuYKCApmEjYiIiKg0TMSU1LJlS/z333/Yvn07bt68iV9++QUikQidOnWSKXf79m3UqVNHTVESERFRVaKl7gCqivHjx+PkyZNYsWIFAEAQBLRt2xYtWrSQlklOTsadO3cwaNAgdYVJREREVQgT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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 360x360 with 1 Axes>"
            ],
            "image/png": "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oKOjg++//16FlRIREfVdDDpKKikpwbhx42Bvb99hH3t7e4wbNw7FxcW9WBkRERFJMOgoqaWlBSKR6KH9RCIRWltbe6EiIiIiuh+DjpJsbGyQm5uL5ubmDvs0NzcjLy8PNjY2vVgZERERSTDoKOnZZ59FbW0tIiIiFN5luLS0FBEREaitre3URGQiIiLqPrzqSknBwcFITU3FiRMn4OvrCycnJ1haWgJovzz87NmzuHv3LkaNGoUFCxaouFp63HT2knRejk5EpBwGHSX1798fu3btwvr16/Hjjz+isLAQhYWFMu3z5s3D8uXLoaurq8JK6XHU2UvSeTk6EZFyGHS6QF9fH++++y4iIyNRVFSE6upqAIC5uTmcnZ0fabIyERER9RwGnW4gEong7u6u6jKI6P8pc8diDg8SqScGHSU1NTWhoqIC5ubmGDRokMI+tbW1qK6uho2NDR+rQNSLlH3YKIcHidQPg46Sdu7ciejoaPzwww8dBp2KigrMmzcPS5cuxaJFi3q5QurreFajc5R5ZhkRPf4YdJSUlpYGGxsbuLi4dNjHxcUFNjY2SE1NZdChXsezGp2j7DPLiOjxxqCjpMrKSowePfqh/YYOHYpTp071QkVE6kuZs1NERACDjtJu3bqF/v37P7Sfrq7uA++eTEQPx6eEE5GyeGdkJVlYWKCgoOCBfQRBQGFhIczNzXupKiIiIroXg46SJk2ahMuXL2PHjh0d9vn6669RVVWFSZMm9WJlREREJMGhKyWFhoYiISEBn332GYqKiuDv7w97e3sAQFlZGeLj43HgwAEYGBggNDRUxdUSERH1TQw6SrKwsMCXX36Jv//97zhw4ACSk5Nl2gVBwMCBA7Fx40bpM7CIiIiodzHodIG7uzt++eUXxMbGIjs7G1euXAEADB48GOPHj0dAQACMjIxUXCWpg966xwvvJUNE6oZBp4uMjIwQFhaGsLAwVZdCaqy37vGizMNGiYgeZ5yMTERERGqLQYeIiIjUFoMOERERqS0GHSIiIlJbDDpERESkthh0iIiISG0x6BAREZHa4n10iKhX8aaERNSbGHSIqFf11s0PiYgADl0RERGRGmPQISIiIrXFoENERERqi0GHiIiI1BaDDhEREaktBh0iIiJSWww6REREpLYYdIiIiEhtMegQERGR2mLQISIiIrXFoENERERqi0GHiIiI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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "* There is a spike for polarity score of 1 for the distribution with Editors Selection, indicative that the editors have a higher preference to positive comments.  \n",
        "* Similary a spike is seen for zero i.e neutral comments that are not Editors Selection."
      ],
      "metadata": {
        "id": "k7NLWaBZQbpw"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Predictive Modeling\n",
        "Predicting whether a comment will be editor's pick using feature editorsSelection as the target variable\n",
        "\n",
        "reference: https://www.kaggle.com/code/aashita/predicting-nyt-s-pick"
      ],
      "metadata": {
        "id": "BfvUvDSzHdV6"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install -U spacy\n",
        "!python -m spacy download en_core_web_sm"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "maL3iIkzTw09",
        "outputId": "bb09a197-8f92-40c3-9f44-480095d212f0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: spacy in /usr/local/lib/python3.7/dist-packages (3.2.4)\n",
            "Requirement already satisfied: thinc<8.1.0,>=8.0.12 in /usr/local/lib/python3.7/dist-packages (from spacy) (8.0.15)\n",
            "Requirement already satisfied: pathy>=0.3.5 in /usr/local/lib/python3.7/dist-packages (from spacy) (0.6.1)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (21.3)\n",
            "Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.0.6)\n",
            "Requirement already satisfied: click<8.1.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (7.1.2)\n",
            "Requirement already satisfied: pydantic!=1.8,!=1.8.1,<1.9.0,>=1.7.4 in /usr/local/lib/python3.7/dist-packages (from spacy) (1.8.2)\n",
            "Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (1.21.5)\n",
            "Requirement already satisfied: srsly<3.0.0,>=2.4.1 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.4.2)\n",
            "Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.0.6)\n",
            "Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy) (3.0.6)\n",
            "Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (1.0.6)\n",
            "Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (1.0.2)\n",
            "Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (3.3.0)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.11.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from spacy) (57.4.0)\n",
            "Requirement already satisfied: typer<0.5.0,>=0.3.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (0.4.1)\n",
            "Requirement already satisfied: wasabi<1.1.0,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from spacy) (0.9.0)\n",
            "Requirement already satisfied: blis<0.8.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (0.4.1)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (2.23.0)\n",
            "Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.7/dist-packages (from spacy) (4.63.0)\n",
            "Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.8 in /usr/local/lib/python3.7/dist-packages (from spacy) (3.0.9)\n",
            "Requirement already satisfied: typing-extensions<4.0.0.0,>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from spacy) (3.10.0.2)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from catalogue<2.1.0,>=2.0.6->spacy) (3.7.0)\n",
            "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->spacy) (3.0.7)\n",
            "Requirement already satisfied: smart-open<6.0.0,>=5.0.0 in /usr/local/lib/python3.7/dist-packages (from pathy>=0.3.5->spacy) (5.2.1)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (3.0.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2021.10.8)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (1.24.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2.10)\n",
            "Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2->spacy) (2.0.1)\n",
            "Collecting en-core-web-sm==3.2.0\n",
            "  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl (13.9 MB)\n",
            "\u001b[K     |████████████████████████████████| 13.9 MB 4.3 MB/s \n",
            "\u001b[?25hRequirement already satisfied: spacy<3.3.0,>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from en-core-web-sm==3.2.0) (3.2.4)\n",
            "Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.8 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (3.0.9)\n",
            "Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (1.0.2)\n",
            "Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (1.0.6)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (21.3)\n",
            "Requirement already satisfied: typer<0.5.0,>=0.3.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (0.4.1)\n",
            "Requirement already satisfied: click<8.1.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (7.1.2)\n",
            "Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (3.3.0)\n",
            "Requirement already satisfied: blis<0.8.0,>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (0.4.1)\n",
            "Requirement already satisfied: pathy>=0.3.5 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (0.6.1)\n",
            "Requirement already satisfied: srsly<3.0.0,>=2.4.1 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (2.4.2)\n",
            "Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (2.0.6)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (2.11.3)\n",
            "Requirement already satisfied: wasabi<1.1.0,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (0.9.0)\n",
            "Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (1.21.5)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (2.23.0)\n",
            "Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (3.0.6)\n",
            "Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (2.0.6)\n",
            "Requirement already satisfied: thinc<8.1.0,>=8.0.12 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (8.0.15)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (57.4.0)\n",
            "Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (4.63.0)\n",
            "Requirement already satisfied: pydantic!=1.8,!=1.8.1,<1.9.0,>=1.7.4 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (1.8.2)\n",
            "Requirement already satisfied: typing-extensions<4.0.0.0,>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (3.10.0.2)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from catalogue<2.1.0,>=2.0.6->spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (3.7.0)\n",
            "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (3.0.7)\n",
            "Requirement already satisfied: smart-open<6.0.0,>=5.0.0 in /usr/local/lib/python3.7/dist-packages (from pathy>=0.3.5->spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (5.2.1)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (2.10)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (3.0.4)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (1.24.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0,>=2.13.0->spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (2021.10.8)\n",
            "Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2->spacy<3.3.0,>=3.2.0->en-core-web-sm==3.2.0) (2.0.1)\n",
            "\u001b[38;5;2m✔ Download and installation successful\u001b[0m\n",
            "You can now load the package via spacy.load('en_core_web_sm')\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#using df_comments dataset\n",
        "#from pre-processing steps in the EDA above there are no null values in the commentsBody\n",
        "\n",
        "import pandas as pd \n",
        "\n",
        "\n",
        "df_art_jan17 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/ArticlesJan2017.csv')\n",
        "df_art_jan18 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/ArticlesJan2018.csv')\n",
        "df_art_feb17 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/ArticlesFeb2017.csv')\n",
        "df_art_feb18 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/ArticlesFeb2018.csv')\n",
        "df_art_mar17 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/ArticlesMarch2017.csv')\n",
        "df_art_mar18 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/ArticlesMarch2018.csv')\n",
        "df_art_apr17 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/ArticlesApril2017.csv')\n",
        "df_art_apr18 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/ArticlesApril2018.csv')\n",
        "\n",
        "\n",
        "df_jan17 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/CommentsJan2017.csv')\n",
        "df_jan18 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/CommentsJan2018.csv')\n",
        "df_feb17 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/CommentsFeb2017.csv')\n",
        "df_feb18 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/CommentsFeb2018.csv')\n",
        "df_mar17 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/CommentsMarch2017.csv')\n",
        "df_mar18 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/CommentsMarch2018.csv')\n",
        "df_apr17 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/CommentsApril2017.csv')\n",
        "df_apr18 = pd.read_csv('/content/drive/Shareddrives/Data Science with Python/Datasets/comments/CommentsApril2018.csv')\n",
        "\n",
        "comment = [df_jan17, df_jan18, df_feb17, df_feb18, df_mar17, df_mar18,  df_apr17, df_apr18]\n",
        "df_com = pd.concat(comment)\n",
        "\n",
        "articles = [df_art_jan17, df_art_feb17, df_art_mar17, df_art_apr17, df_art_jan18, df_art_feb18, df_art_mar18, df_art_apr18]\n",
        "df_art = pd.concat(articles)\n",
        "\n",
        "df_merge = pd.merge(df_art, df_com, on='articleID', how='inner') "
      ],
      "metadata": {
        "id": "bjcMdoflHWZ6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1387420b-5c11-4c92-a7eb-8043fd51369a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2882: DtypeWarning: Columns (14,15,31,32) have mixed types.Specify dtype option on import or set low_memory=False.\n",
            "  exec(code_obj, self.user_global_ns, self.user_ns)\n",
            "/usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2882: DtypeWarning: Columns (32) have mixed types.Specify dtype option on import or set low_memory=False.\n",
            "  exec(code_obj, self.user_global_ns, self.user_ns)\n",
            "/usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2882: DtypeWarning: Columns (32,33) have mixed types.Specify dtype option on import or set low_memory=False.\n",
            "  exec(code_obj, self.user_global_ns, self.user_ns)\n",
            "/usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2882: DtypeWarning: Columns (14,15,31) have mixed types.Specify dtype option on import or set low_memory=False.\n",
            "  exec(code_obj, self.user_global_ns, self.user_ns)\n",
            "/usr/local/lib/python3.7/dist-packages/IPython/core/interactiveshell.py:2882: DtypeWarning: Columns (25,26) have mixed types.Specify dtype option on import or set low_memory=False.\n",
            "  exec(code_obj, self.user_global_ns, self.user_ns)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_merge.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-wP0MBDsEV9-",
        "outputId": "3b9087aa-feaa-483c-88a8-b1674e052152"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1878045, 49)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_merge.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e6T3YeSLeWA-",
        "outputId": "3ddaf022-fc06-408d-db4f-0fe126edb232"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "articleID                    0\n",
              "abstract                 71583\n",
              "byline                       0\n",
              "documentType                 0\n",
              "headline                     0\n",
              "keywords                     0\n",
              "multimedia                   0\n",
              "newDesk_x                    0\n",
              "printPage_x                  0\n",
              "pubDate                      0\n",
              "sectionName_x                0\n",
              "snippet                      0\n",
              "source                       0\n",
              "typeOfMaterial_x             0\n",
              "webURL                       0\n",
              "articleWordCount_x           0\n",
              "approveDate                  0\n",
              "articleWordCount_y           0\n",
              "commentBody                  0\n",
              "commentID                    0\n",
              "commentSequence              0\n",
              "commentTitle              2179\n",
              "commentType                  0\n",
              "createDate                   0\n",
              "depth                        0\n",
              "editorsSelection             0\n",
              "inReplyTo                    0\n",
              "newDesk_y                    0\n",
              "parentID                     0\n",
              "parentUserDisplayName    61164\n",
              "permID                       0\n",
              "picURL                       0\n",
              "printPage_y                  0\n",
              "recommendations              0\n",
              "recommendedFlag          71621\n",
              "replyCount                   0\n",
              "reportAbuseFlag          71621\n",
              "sectionName_y             6046\n",
              "sharing                      0\n",
              "status                       0\n",
              "timespeople                  0\n",
              "trusted                      0\n",
              "updateDate                   0\n",
              "userDisplayName             34\n",
              "userID                       0\n",
              "userLocation                19\n",
              "userTitle                71606\n",
              "userURL                  71621\n",
              "typeOfMaterial_y             0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 118
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#check imbalance in dataset\n",
        "import seaborn as sns\n",
        "sns.countplot(x= 'editorsSelection', data = df_merge)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 307
        },
        "id": "0bL1eIct0w9U",
        "outputId": "593b7274-87d7-44d9-da8a-86f7e5190f38"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f2d2fa905d0>"
            ]
          },
          "metadata": {},
          "execution_count": 6
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print((float(pd.value_counts(df_merge['editorsSelection'])[1])/float(pd.value_counts(df_merge['editorsSelection']).sum())) * 100)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zbqyCR_pus9b",
        "outputId": "320a779e-a565-4321-b5ca-c14dfe797362"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1.9071428000926496\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "minority class 1 is only 1.9% of the entire \n",
        "dataset"
      ],
      "metadata": {
        "id": "tZAIO4B2u2bc"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "We can clearly see that there is a class imbalance here with very few articles being picked as Editors Selection and a majority of them of not. We will need to handle this class imbalance prior to predictive modeling and this could be done by undersampling the majority class. \n"
      ],
      "metadata": {
        "id": "xWJk67Y-3f6N"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "####Undersamplig the majority class\n",
        "First we discard all the comments from articles that have no comments picked as Editor's selection. From the remaining articles, we randomly pick comments from the majority class so as to have a ratio of 3:1."
      ],
      "metadata": {
        "id": "X1s3-sQU5dXg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "ratio = 1\n",
        "def balance_classes(group):\n",
        "    picked = group.loc[group.editorsSelection == True]\n",
        "    n = round(picked.shape[0]*ratio)\n",
        "    if n:        \n",
        "        try:\n",
        "            not_picked = group.loc[group.editorsSelection == False].sample(n)\n",
        "        except: # In case, fewer than n comments with `editorsSelection == False`\n",
        "            not_picked = group.loc[group.editorsSelection == False]\n",
        "        balanced_group = pd.concat([picked, not_picked])\n",
        "        return balanced_group\n",
        "    else: # If no editor's pick for an article, dicard all comments from that article\n",
        "        return None \n",
        "\n",
        "#New balanced dataset \n",
        "df_merge = df_merge.groupby('articleID').apply(balance_classes).reset_index(drop=True)"
      ],
      "metadata": {
        "id": "SkEqfhQs3fUj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_merge.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4LjMWrqF6DPr",
        "outputId": "3bfdbf42-d96f-4cd7-8eff-f8e5b74aff32"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(71621, 49)"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#check imbalance for new dataset\n",
        "import seaborn as sns\n",
        "sns.countplot(x= 'editorsSelection', data = df_merge)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "wIgG_uXT3KR4",
        "outputId": "fad2e5a2-e864-4ab4-ec9b-bc62112bcb07"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f2d2e52ee50>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print((float(pd.value_counts(df_merge['editorsSelection'])[1])/float(pd.value_counts(df_merge['editorsSelection']).sum())) * 100)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5uwayGg0vTZI",
        "outputId": "3f5020ac-f4e0-4d8e-b4c5-120434ad8261"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "50.00907555046704\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Now editorsSelection class = 1 is balanced to represnts 25% of the dataset. "
      ],
      "metadata": {
        "id": "VrT95h7WvZ5l"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "For the modeling we would need only three features articleID, commentBody and editorsSelection, lets drop other columns.\n"
      ],
      "metadata": {
        "id": "2YPbsQis6WfE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Date prep/pre-processing based for balanced dataset. "
      ],
      "metadata": {
        "id": "sjWcwOkRPfxu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import spacy\n",
        "from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer\n",
        "from sklearn.base import TransformerMixin\n",
        "from sklearn.pipeline import Pipeline"
      ],
      "metadata": {
        "id": "rzExeTweRdIM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Tokenizing data with spacy\n",
        "import string\n",
        "#from spacy.lang.en.stop_words import STOP_WORDS\n",
        "from spacy.lang.en import English\n",
        "\n",
        "# Create our list of punctuation marks\n",
        "punctuations = string.punctuation\n",
        "\n",
        "# Create our list of stopwords\n",
        "nlp = spacy.load('en_core_web_sm')\n",
        "stop_words = spacy.lang.en.stop_words.STOP_WORDS\n",
        "\n",
        "# Load English tokenizer, tagger, parser, NER and word vectors\n",
        "parser = English()\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "# Creating our tokenizer function\n",
        "def spacy_tokenizer(sentence):\n",
        "    # Creating our token object, which is used to create documents with linguistic annotations.\n",
        "    mytokens = parser(sentence)\n",
        "\n",
        "    # Lemmatizing each token and converting each token into lowercase\n",
        "    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != \"-PRON-\" else word.lower_ for word in mytokens ]\n",
        "    #mytokens = [word.porter.stem() for word in mytokens]\n",
        "    #mytokens= [word for word.lower in my tokens] #lowercase \n",
        "\n",
        "    # Removing stop words\n",
        "    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]\n",
        "\n",
        "    # return preprocessed list of tokens\n",
        "    return mytokens\n"
      ],
      "metadata": {
        "id": "StgnmFzv_ymD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Custom transformer using spaCy\n",
        "class predictors(TransformerMixin):\n",
        "    def transform(self, X, **transform_params):\n",
        "        # Cleaning Text\n",
        "        return [clean_text(text) for text in X]\n",
        "\n",
        "    def fit(self, X, y=None, **fit_params):\n",
        "        return self\n",
        "\n",
        "    def get_params(self, deep=True):\n",
        "        return {}\n",
        "\n",
        "# Basic function to clean the text\n",
        "def clean_text(text):\n",
        "    # Removing spaces and converting text into lowercase\n",
        "    return text.strip().lower()\n"
      ],
      "metadata": {
        "id": "MHI-NtUD_yjf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Vectorization Feature Engineering - TF-IDF\n"
      ],
      "metadata": {
        "id": "4sbwcxxz_y_K"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#using unigrams\n",
        "#bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))"
      ],
      "metadata": {
        "id": "qN6XFNYLUGI-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#tf-idf\n",
        "tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)\n",
        "#from sklearn.preprocessing import StandardScaler\n",
        "#scaler = StandardScaler()\n",
        "#tfidf_vector = scaler.fit_transform(tfidf_vector)"
      ],
      "metadata": {
        "id": "WSOBl2gsUOoI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X = df_merge['commentBody'] # the features we want to analyze\n",
        "y = df_merge['editorsSelection'] # the labels, or answers, we want to test against\n",
        "y = y.astype('int')\n",
        "\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)"
      ],
      "metadata": {
        "id": "nx_63R91Uv6O"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DDsELswTV2g5",
        "outputId": "749bf214-9e44-48b2-8996-e1c413473ade"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0        Of course Obama's policies are at risk.  As Ob...\n",
              "1        As someone with a son who has cystic fibrosis ...\n",
              "2        I really can't see how some people complain th...\n",
              "3        If you listen to what the Republicans are sayi...\n",
              "4        The \"repeal and delay\" strategy is a cynical a...\n",
              "                               ...                        \n",
              "71616    I imagine I am not the only one who sees Muell...\n",
              "71617    To those who think Mueller’s team leaked: any ...\n",
              "71618    There's nothing surprising here, just an obvio...\n",
              "71619    \"That document was provided to The Times by a ...\n",
              "71620    Robert Mueller must be livid that his ultra se...\n",
              "Name: commentBody, Length: 71621, dtype: object"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Create Pipeline and Generate the Model"
      ],
      "metadata": {
        "id": "t9HM3MraVzbx"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Logistic Regression Classifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "classifier = LogisticRegression()\n",
        "\n",
        "# Create pipeline using tfidfvector\n",
        "pipe = Pipeline([(\"cleaner\", predictors()),\n",
        "                 ('vectorizer', tfidf_vector),\n",
        "                 ('classifier', classifier)])\n",
        "\n",
        "# model generation\n",
        "pipe.fit(X_train, y_train)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o9Qi3_bnVyu2",
        "outputId": "4edc319a-e89a-4fa8-a26b-f87525d4d840"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Pipeline(steps=[('cleaner', <__main__.predictors object at 0x7f2cffcc3d10>),\n",
              "                ('vectorizer',\n",
              "                 TfidfVectorizer(tokenizer=<function spacy_tokenizer at 0x7f2d2e950e60>)),\n",
              "                ('classifier', LogisticRegression())])"
            ]
          },
          "metadata": {},
          "execution_count": 103
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "\n",
        "predicted = pipe.predict(X_test)\n",
        "\n",
        "# Model Accuracy\n",
        "print(\"Accuracy:\",metrics.accuracy_score(y_test, predicted))\n",
        "print(\"Precision:\",metrics.precision_score(y_test, predicted))\n",
        "print(\"Recall:\",metrics.recall_score(y_test, predicted))\n",
        "print(\"f1 score\",metrics.f1_score(y_test, predicted))\n",
        "\n",
        "Accuracy_lr = metrics.accuracy_score(y_test, predicted)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ek_kCuQufODW",
        "outputId": "ad84314d-d6bf-4b26-f1ed-a2b0d9341106"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.6513240564061991\n",
            "Precision: 0.6499819624819625\n",
            "Recall: 0.6662044740247736\n",
            "f1 score 0.6579932438601296\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
        "import matplotlib.pyplot as plt\n",
        "#Confusion Matrix\n",
        "cm = confusion_matrix(y_true=y_test, y_pred=predicted)\n",
        "labels= y.keys()\n",
        "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)\n",
        "disp.plot()\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "1CgssjSvx7bM",
        "outputId": "457e6973-1204-4521-960f-fec9c9589dbd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_test.value_counts()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2vwTb7xmc3F0",
        "outputId": "517d9879-1ae8-4164-b194-4edf8f4f2a69"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1    10818\n",
              "0    10669\n",
              "Name: editorsSelection, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#feature names/tags\n",
        "\n",
        "terms = tfidf_vector.get_feature_names_out()\n",
        "\n",
        "X_traintfidf = tfidf_vector.fit_transform(X_train)\n",
        "\n",
        "# sum tfidf frequency of each term through documents\n",
        "sums = X_traintfidf.sum(axis=0)\n",
        "\n",
        "# connecting term to its sums frequency\n",
        "data = []\n",
        "for col, term in enumerate(terms):\n",
        "    data.append( (term, sums[0,col] ))\n",
        "\n",
        "ranking = pd.DataFrame(data, columns=['term','rank'])\n",
        "print(ranking.sort_values('rank', ascending=False))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mDE2DcDUB94V",
        "outputId": "dd5dfc2f-95c7-4a16-b092-d63c3fa4bafb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              term         rank\n",
            "89494        trump  1359.327731\n",
            "65320       people   871.719362\n",
            "52902         like   646.359836\n",
            "68531    president   577.611418\n",
            "87272         time   493.550235\n",
            "...            ...          ...\n",
            "15715  br/>centene     0.063746\n",
            "2646         28.09     0.063746\n",
            "8006      antigens     0.057157\n",
            "41092         h3n2     0.057157\n",
            "67415  pomegranate     0.057157\n",
            "\n",
            "[97851 rows x 2 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "tfidf_vector.get_feature_names()"
      ],
      "metadata": {
        "id": "02AlZacqOJ6R",
        "outputId": "16e5bb5e-4e0a-44ef-d42f-87bfc6d56f9d",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.\n",
            "  warnings.warn(msg, category=FutureWarning)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['!!!<br/>this',\n",
              " '!!<br/><br/>nabokov',\n",
              " '!!<br/><br/>stan',\n",
              " '!!<br/>build',\n",
              " \"!!<br/>don't\",\n",
              " '!<br/><br/>this',\n",
              " '\"\"<br/><br/>did',\n",
              " '\"\"<br/><br/>the',\n",
              " '\"\\'<br/><br/>complete',\n",
              " '\"(washington',\n",
              " '\")',\n",
              " '\").<br/><br/>government',\n",
              " '\")<br/><br/>it',\n",
              " '\")<br/><br/>usually',\n",
              " '\")<br/><br/>when',\n",
              " '\")<br/>the',\n",
              " '\")but',\n",
              " '\")many',\n",
              " '\"-',\n",
              " '\"---that',\n",
              " '\"--benjamin',\n",
              " '\"--even',\n",
              " '\"--i',\n",
              " '\"--jean',\n",
              " '\"--what',\n",
              " '\"-woody',\n",
              " '\".<br/><br/>she',\n",
              " '\"<br/',\n",
              " '\"<br/>\"a',\n",
              " '\"<br/>\"hey',\n",
              " '\"<br/>\"i',\n",
              " '\"<br/>\"like',\n",
              " '\"<br/>\"make',\n",
              " '\"<br/>\"propoganda',\n",
              " '\"<br/>\"put',\n",
              " '\"<br/>\"say',\n",
              " '\"<br/>\"the',\n",
              " '\"<br/>\"uh',\n",
              " '\"<br/>\"we\\'re',\n",
              " '\"<br/>$1,500',\n",
              " '\"<br/>(fantastic',\n",
              " '\"<br/>-',\n",
              " '\"<br/>--',\n",
              " '\"<br/>---------------------',\n",
              " '\"<br/>-------<br/>aren\\'t',\n",
              " '\"<br/>--donald',\n",
              " '\"<br/>--this',\n",
              " '\"<br/>--true',\n",
              " '\"<br/>-<br/>that',\n",
              " '\"<br/>-ayn',\n",
              " '\"<br/>-euripedes',\n",
              " '\"<br/>-gerald',\n",
              " '\"<br/>-hare',\n",
              " '\"<br/>.<br/>i',\n",
              " '\"<br/>2016',\n",
              " '\"<br/>98',\n",
              " '\"<br/><a',\n",
              " '\"<br/><br/',\n",
              " '\"<br/><br/>',\n",
              " '\"<br/><br/>\"all',\n",
              " '\"<br/><br/>\"approximately',\n",
              " '\"<br/><br/>\"but',\n",
              " '\"<br/><br/>\"contradictionary',\n",
              " '\"<br/><br/>\"freedom',\n",
              " '\"<br/><br/>\"i',\n",
              " '\"<br/><br/>\"mr',\n",
              " '\"<br/><br/>\"neri',\n",
              " '\"<br/><br/>\"no',\n",
              " '\"<br/><br/>\"philosophy',\n",
              " '\"<br/><br/>\"republican',\n",
              " '\"<br/><br/>\"show',\n",
              " '\"<br/><br/>\"stuff',\n",
              " '\"<br/><br/>\"the',\n",
              " '\"<br/><br/>\"thus',\n",
              " '\"<br/><br/>\"we',\n",
              " '\"<br/><br/>\"we\\'ll',\n",
              " '\"<br/><br/>\"whatever',\n",
              " '\"<br/><br/>\"yes',\n",
              " '\"<br/><br/>\"you',\n",
              " '\"<br/><br/>\\'it',\n",
              " '\"<br/><br/>(cnn',\n",
              " '\"<br/><br/>-',\n",
              " '\"<br/><br/>---',\n",
              " '\"<br/><br/>----',\n",
              " '\"<br/><br/>10',\n",
              " '\"<br/><br/>3',\n",
              " '\"<br/><br/>5',\n",
              " '\"<br/><br/>50',\n",
              " '\"<br/><br/>54b$',\n",
              " '\"<br/><br/><a',\n",
              " '\"<br/><br/>?????this',\n",
              " '\"<br/><br/>@realdonaldtrump',\n",
              " '\"<br/><br/>a',\n",
              " '\"<br/><br/>a.k.a',\n",
              " '\"<br/><br/>about',\n",
              " '\"<br/><br/>abraham',\n",
              " '\"<br/><br/>absolutely',\n",
              " '\"<br/><br/>actually',\n",
              " '\"<br/><br/>adjusting',\n",
              " '\"<br/><br/>after',\n",
              " '\"<br/><br/>agreed',\n",
              " '\"<br/><br/>all',\n",
              " '\"<br/><br/>allow',\n",
              " '\"<br/><br/>also',\n",
              " '\"<br/><br/>amen',\n",
              " '\"<br/><br/>american',\n",
              " '\"<br/><br/>americans',\n",
              " '\"<br/><br/>an',\n",
              " '\"<br/><br/>and',\n",
              " '\"<br/><br/>another',\n",
              " '\"<br/><br/>any',\n",
              " '\"<br/><br/>anybody',\n",
              " '\"<br/><br/>anyone',\n",
              " '\"<br/><br/>apart',\n",
              " '\"<br/><br/>apparently',\n",
              " '\"<br/><br/>are',\n",
              " '\"<br/><br/>as',\n",
              " '\"<br/><br/>ashis',\n",
              " '\"<br/><br/>assad',\n",
              " '\"<br/><br/>at',\n",
              " '\"<br/><br/>author',\n",
              " '\"<br/><br/>aw',\n",
              " '\"<br/><br/>back',\n",
              " '\"<br/><br/>baltimore',\n",
              " '\"<br/><br/>bannon',\n",
              " '\"<br/><br/>bart',\n",
              " '\"<br/><br/>basically',\n",
              " '\"<br/><br/>beautiful',\n",
              " '\"<br/><br/>beautifully',\n",
              " '\"<br/><br/>before',\n",
              " '\"<br/><br/>ben',\n",
              " '\"<br/><br/>bingo',\n",
              " '\"<br/><br/>bingo!<br/>do',\n",
              " '\"<br/><br/>bipolar',\n",
              " '\"<br/><br/>block',\n",
              " '\"<br/><br/>brian',\n",
              " '\"<br/><br/>brilliant',\n",
              " '\"<br/><br/>but',\n",
              " '\"<br/><br/>by',\n",
              " '\"<br/><br/>c\\'mon',\n",
              " '\"<br/><br/>can',\n",
              " '\"<br/><br/>capitalism',\n",
              " '\"<br/><br/>carlon',\n",
              " '\"<br/><br/>case',\n",
              " '\"<br/><br/>change',\n",
              " '\"<br/><br/>charles',\n",
              " '\"<br/><br/>charlie',\n",
              " '\"<br/><br/>china',\n",
              " '\"<br/><br/>christie',\n",
              " '\"<br/><br/>colorful',\n",
              " '\"<br/><br/>come',\n",
              " '\"<br/><br/>comey',\n",
              " '\"<br/><br/>comey:<br/>\"did',\n",
              " '\"<br/><br/>comey:<br/>\"good',\n",
              " '\"<br/><br/>commentary',\n",
              " '\"<br/><br/>compared',\n",
              " '\"<br/><br/>considering',\n",
              " '\"<br/><br/>coretta',\n",
              " '\"<br/><br/>correct',\n",
              " '\"<br/><br/>cruz',\n",
              " '\"<br/><br/>cue',\n",
              " '\"<br/><br/>dan',\n",
              " '\"<br/><br/>david',\n",
              " '\"<br/><br/>democrats',\n",
              " '\"<br/><br/>disruption',\n",
              " '\"<br/><br/>do',\n",
              " '\"<br/><br/>doctors',\n",
              " '\"<br/><br/>does',\n",
              " '\"<br/><br/>donald',\n",
              " '\"<br/><br/>dr',\n",
              " '\"<br/><br/>duh',\n",
              " '\"<br/><br/>during',\n",
              " '\"<br/><br/>editorial',\n",
              " '\"<br/><br/>either',\n",
              " '\"<br/><br/>elena',\n",
              " '\"<br/><br/>elsewhere',\n",
              " '\"<br/><br/>enough',\n",
              " '\"<br/><br/>equally',\n",
              " '\"<br/><br/>eu',\n",
              " '\"<br/><br/>even',\n",
              " '\"<br/><br/>ever',\n",
              " '\"<br/><br/>every',\n",
              " '\"<br/><br/>exactly',\n",
              " '\"<br/><br/>excellent.<br/>i',\n",
              " '\"<br/><br/>except',\n",
              " '\"<br/><br/>executive',\n",
              " '\"<br/><br/>expletive',\n",
              " '\"<br/><br/>fact',\n",
              " '\"<br/><br/>firing',\n",
              " '\"<br/><br/>first',\n",
              " '\"<br/><br/>flakiest',\n",
              " '\"<br/><br/>folks',\n",
              " '\"<br/><br/>for',\n",
              " '\"<br/><br/>forgive',\n",
              " '\"<br/><br/>frank',\n",
              " '\"<br/><br/>frankly',\n",
              " '\"<br/><br/>frighting',\n",
              " '\"<br/><br/>from',\n",
              " '\"<br/><br/>furthermore',\n",
              " '\"<br/><br/>gail',\n",
              " '\"<br/><br/>give',\n",
              " '\"<br/><br/>god',\n",
              " '\"<br/><br/>good',\n",
              " '\"<br/><br/>great',\n",
              " '\"<br/><br/>guilt',\n",
              " '\"<br/><br/>ha',\n",
              " '\"<br/><br/>has',\n",
              " '\"<br/><br/>have',\n",
              " '\"<br/><br/>having',\n",
              " '\"<br/><br/>hawking',\n",
              " '\"<br/><br/>he',\n",
              " '\"<br/><br/>her',\n",
              " '\"<br/><br/>here',\n",
              " '\"<br/><br/>hmmm',\n",
              " '\"<br/><br/>holy',\n",
              " '\"<br/><br/>how',\n",
              " '\"<br/><br/>however',\n",
              " '\"<br/><br/>huh',\n",
              " '\"<br/><br/>humphrey',\n",
              " '\"<br/><br/>i',\n",
              " '\"<br/><br/>i\\'ll',\n",
              " '\"<br/><br/>i\\'m',\n",
              " '\"<br/><br/>i\\'ve',\n",
              " '\"<br/><br/>identity',\n",
              " '\"<br/><br/>if',\n",
              " '\"<br/><br/>impeachment',\n",
              " '\"<br/><br/>in',\n",
              " '\"<br/><br/>including',\n",
              " '\"<br/><br/>indeed',\n",
              " '\"<br/><br/>instead',\n",
              " '\"<br/><br/>is',\n",
              " '\"<br/><br/>isn\\'t',\n",
              " '\"<br/><br/>it',\n",
              " '\"<br/><br/>jackson',\n",
              " '\"<br/><br/>james',\n",
              " '\"<br/><br/>jeff',\n",
              " '\"<br/><br/>jim',\n",
              " '\"<br/><br/>johnson',\n",
              " '\"<br/><br/>just',\n",
              " '\"<br/><br/>keep',\n",
              " '\"<br/><br/>kinda',\n",
              " '\"<br/><br/>kudlow',\n",
              " '\"<br/><br/>lady',\n",
              " '\"<br/><br/>last',\n",
              " '\"<br/><br/>later',\n",
              " '\"<br/><br/>let',\n",
              " '\"<br/><br/>letting',\n",
              " '\"<br/><br/>lewis',\n",
              " '\"<br/><br/>liberals',\n",
              " '\"<br/><br/>little',\n",
              " '\"<br/><br/>lol',\n",
              " '\"<br/><br/>look',\n",
              " '\"<br/><br/>looks',\n",
              " '\"<br/><br/>love',\n",
              " '\"<br/><br/>ludicrous',\n",
              " '\"<br/><br/>maddening',\n",
              " '\"<br/><br/>makes',\n",
              " '\"<br/><br/>many',\n",
              " '\"<br/><br/>mark',\n",
              " '\"<br/><br/>maureen',\n",
              " '\"<br/><br/>may',\n",
              " '\"<br/><br/>maybe',\n",
              " '\"<br/><br/>mcconnell',\n",
              " '\"<br/><br/>me',\n",
              " '\"<br/><br/>meanwhile',\n",
              " '\"<br/><br/>men',\n",
              " '\"<br/><br/>millions',\n",
              " '\"<br/><br/>mimi',\n",
              " '\"<br/><br/>ministers',\n",
              " '\"<br/><br/>missing',\n",
              " '\"<br/><br/>more',\n",
              " '\"<br/><br/>most',\n",
              " '\"<br/><br/>mr',\n",
              " '\"<br/><br/>ms',\n",
              " '\"<br/><br/>my',\n",
              " '\"<br/><br/>near',\n",
              " '\"<br/><br/>nearly',\n",
              " '\"<br/><br/>never',\n",
              " '\"<br/><br/>nice',\n",
              " '\"<br/><br/>nick',\n",
              " '\"<br/><br/>no',\n",
              " '\"<br/><br/>nobody',\n",
              " '\"<br/><br/>none',\n",
              " '\"<br/><br/>nonsense',\n",
              " '\"<br/><br/>nope',\n",
              " '\"<br/><br/>nor',\n",
              " '\"<br/><br/>not',\n",
              " '\"<br/><br/>nothing',\n",
              " '\"<br/><br/>notice',\n",
              " '\"<br/><br/>now',\n",
              " '\"<br/><br/>nunes',\n",
              " '\"<br/><br/>obamacare',\n",
              " '\"<br/><br/>of',\n",
              " '\"<br/><br/>offshoring',\n",
              " '\"<br/><br/>oh',\n",
              " '\"<br/><br/>okay',\n",
              " '\"<br/><br/>on',\n",
              " '\"<br/><br/>once',\n",
              " '\"<br/><br/>one',\n",
              " '\"<br/><br/>only',\n",
              " '\"<br/><br/>or',\n",
              " '\"<br/><br/>other',\n",
              " '\"<br/><br/>others',\n",
              " '\"<br/><br/>otherwise',\n",
              " '\"<br/><br/>our',\n",
              " '\"<br/><br/>part',\n",
              " '\"<br/><br/>peirce',\n",
              " '\"<br/><br/>people',\n",
              " '\"<br/><br/>perhaps',\n",
              " '\"<br/><br/>phil',\n",
              " '\"<br/><br/>pop',\n",
              " '\"<br/><br/>potus',\n",
              " '\"<br/><br/>president',\n",
              " '\"<br/><br/>progressives',\n",
              " '\"<br/><br/>public',\n",
              " '\"<br/><br/>putin:<br/>\"sure',\n",
              " '\"<br/><br/>quick',\n",
              " '\"<br/><br/>quite',\n",
              " '\"<br/><br/>racing',\n",
              " '\"<br/><br/>rarely',\n",
              " '\"<br/><br/>read',\n",
              " '\"<br/><br/>really',\n",
              " '\"<br/><br/>recycling',\n",
              " '\"<br/><br/>reduce',\n",
              " '\"<br/><br/>reminds',\n",
              " '\"<br/><br/>repairing',\n",
              " '\"<br/><br/>reporter',\n",
              " '\"<br/><br/>republicans',\n",
              " '\"<br/><br/>ricky',\n",
              " '\"<br/><br/>right',\n",
              " '\"<br/><br/>rima',\n",
              " '\"<br/><br/>rip',\n",
              " '\"<br/><br/>ross',\n",
              " '\"<br/><br/>roy',\n",
              " '\"<br/><br/>ryan',\n",
              " '\"<br/><br/>sad',\n",
              " '\"<br/><br/>sadly',\n",
              " '\"<br/><br/>say',\n",
              " '\"<br/><br/>school',\n",
              " '\"<br/><br/>secondary',\n",
              " '\"<br/><br/>secret',\n",
              " '\"<br/><br/>seems',\n",
              " '\"<br/><br/>segregation',\n",
              " '\"<br/><br/>seriously',\n",
              " '\"<br/><br/>shakespeare',\n",
              " '\"<br/><br/>shameful',\n",
              " '\"<br/><br/>she',\n",
              " '\"<br/><br/>should',\n",
              " '\"<br/><br/>simple',\n",
              " '\"<br/><br/>since',\n",
              " '\"<br/><br/>smart',\n",
              " '\"<br/><br/>so',\n",
              " '\"<br/><br/>some',\n",
              " '\"<br/><br/>somebody',\n",
              " '\"<br/><br/>soon',\n",
              " '\"<br/><br/>sooner',\n",
              " '\"<br/><br/>sorry',\n",
              " '\"<br/><br/>sounds',\n",
              " '\"<br/><br/>south',\n",
              " '\"<br/><br/>spare',\n",
              " '\"<br/><br/>steve',\n",
              " '\"<br/><br/>still',\n",
              " '\"<br/><br/>strongly',\n",
              " '\"<br/><br/>surely',\n",
              " '\"<br/><br/>sycophantically',\n",
              " '\"<br/><br/>take',\n",
              " '\"<br/><br/>talk',\n",
              " '\"<br/><br/>tell',\n",
              " '\"<br/><br/>thank',\n",
              " '\"<br/><br/>that',\n",
              " '\"<br/><br/>the',\n",
              " '\"<br/><br/>then',\n",
              " '\"<br/><br/>there',\n",
              " '\"<br/><br/>these',\n",
              " '\"<br/><br/>they',\n",
              " '\"<br/><br/>they\\'re',\n",
              " '\"<br/><br/>think',\n",
              " '\"<br/><br/>this',\n",
              " '\"<br/><br/>those',\n",
              " '\"<br/><br/>through',\n",
              " '\"<br/><br/>throughout',\n",
              " '\"<br/><br/>thus',\n",
              " '\"<br/><br/>time',\n",
              " '\"<br/><br/>to',\n",
              " '\"<br/><br/>today',\n",
              " '\"<br/><br/>tonight',\n",
              " '\"<br/><br/>true',\n",
              " '\"<br/><br/>trump',\n",
              " '\"<br/><br/>try',\n",
              " '\"<br/><br/>tv',\n",
              " '\"<br/><br/>uh',\n",
              " '\"<br/><br/>unfortunately',\n",
              " '\"<br/><br/>unicorn',\n",
              " '\"<br/><br/>unless',\n",
              " '\"<br/><br/>very',\n",
              " '\"<br/><br/>virginia',\n",
              " '\"<br/><br/>wait',\n",
              " '\"<br/><br/>wake',\n",
              " '\"<br/><br/>want',\n",
              " '\"<br/><br/>watching',\n",
              " '\"<br/><br/>way',\n",
              " '\"<br/><br/>we',\n",
              " '\"<br/><br/>we\\'ve',\n",
              " '\"<br/><br/>well',\n",
              " '\"<br/><br/>werd',\n",
              " '\"<br/><br/>what',\n",
              " '\"<br/><br/>when',\n",
              " '\"<br/><br/>where',\n",
              " '\"<br/><br/>whether',\n",
              " '\"<br/><br/>which',\n",
              " '\"<br/><br/>while',\n",
              " '\"<br/><br/>white',\n",
              " '\"<br/><br/>who',\n",
              " '\"<br/><br/>why',\n",
              " '\"<br/><br/>why?<br/><br/>assuming',\n",
              " '\"<br/><br/>winston',\n",
              " '\"<br/><br/>with',\n",
              " '\"<br/><br/>words',\n",
              " '\"<br/><br/>wow',\n",
              " '\"<br/><br/>wrong',\n",
              " '\"<br/><br/>yeah',\n",
              " '\"<br/><br/>yes',\n",
              " '\"<br/><br/>yes.<br/>this',\n",
              " '\"<br/><br/>yesterday',\n",
              " '\"<br/><br/>you',\n",
              " '\"<br/><br/>you\\'re',\n",
              " '\"<br/><br/>you\\'ve',\n",
              " '\"<br/><br/>your',\n",
              " '\"<br/><br/>zuckerberg',\n",
              " '\"<br/><br/>~~<br/><br/>i',\n",
              " '\"<br/><br/>“at',\n",
              " '\"<br/>[d',\n",
              " '\"<br/>________________________<br/><br/>a',\n",
              " '\"<br/>actually',\n",
              " '\"<br/>after',\n",
              " '\"<br/>all',\n",
              " '\"<br/>and',\n",
              " '\"<br/>arthur',\n",
              " '\"<br/>as',\n",
              " '\"<br/>because',\n",
              " '\"<br/>bingo.<br/><br/>s&amp;p',\n",
              " '\"<br/>bravo',\n",
              " '\"<br/>but',\n",
              " '\"<br/>by',\n",
              " '\"<br/>call',\n",
              " '\"<br/>can',\n",
              " '\"<br/>change',\n",
              " '\"<br/>cohen',\n",
              " '\"<br/>deconstructed',\n",
              " '\"<br/>despite',\n",
              " '\"<br/>doesn\\'t',\n",
              " '\"<br/>don\\'t',\n",
              " '\"<br/>duh',\n",
              " '\"<br/>even',\n",
              " '\"<br/>everywhere',\n",
              " '\"<br/>exactimundo',\n",
              " '\"<br/>facts',\n",
              " '\"<br/>false-',\n",
              " '\"<br/>feel',\n",
              " '\"<br/>flash',\n",
              " '\"<br/>for',\n",
              " '\"<br/>fortunately',\n",
              " '\"<br/>good',\n",
              " '\"<br/>gowry',\n",
              " '\"<br/>hasty',\n",
              " '\"<br/>he',\n",
              " '\"<br/>hello?<br/><br/>',\n",
              " '\"<br/>here',\n",
              " '\"<br/>hopefully',\n",
              " '\"<br/>hyper',\n",
              " '\"<br/>i',\n",
              " '\"<br/>i\\'m',\n",
              " '\"<br/>if',\n",
              " '\"<br/>in',\n",
              " '\"<br/>indeed',\n",
              " '\"<br/>indeed.<br/>because',\n",
              " '\"<br/>is',\n",
              " '\"<br/>it',\n",
              " '\"<br/>job',\n",
              " '\"<br/>keep',\n",
              " '\"<br/>later',\n",
              " '\"<br/>let',\n",
              " '\"<br/>like',\n",
              " '\"<br/>ludicrous?<br/>obscene',\n",
              " '\"<br/>macbeth',\n",
              " '\"<br/>make',\n",
              " '\"<br/>martin',\n",
              " '\"<br/>mr',\n",
              " '\"<br/>ms',\n",
              " '\"<br/>my',\n",
              " '\"<br/>no',\n",
              " '\"<br/>none',\n",
              " '\"<br/>not',\n",
              " '\"<br/>nothing',\n",
              " '\"<br/>npr',\n",
              " '\"<br/>often',\n",
              " '\"<br/>one',\n",
              " '\"<br/>op',\n",
              " '\"<br/>osha',\n",
              " '\"<br/>over',\n",
              " '\"<br/>overusing',\n",
              " '\"<br/>perfect',\n",
              " '\"<br/>please',\n",
              " '\"<br/>plus',\n",
              " '\"<br/>poor',\n",
              " '\"<br/>pretty',\n",
              " '\"<br/>problem',\n",
              " '\"<br/>read',\n",
              " '\"<br/>reading',\n",
              " '\"<br/>reality-\"team',\n",
              " '\"<br/>really',\n",
              " '\"<br/>repeating',\n",
              " '\"<br/>republicans',\n",
              " '\"<br/>rest',\n",
              " '\"<br/>roger',\n",
              " '\"<br/>sessions',\n",
              " '\"<br/>sheez',\n",
              " '\"<br/>so',\n",
              " '\"<br/>somehow',\n",
              " '\"<br/>someone',\n",
              " '\"<br/>still',\n",
              " '\"<br/>stop',\n",
              " '\"<br/>such',\n",
              " '\"<br/>that',\n",
              " '\"<br/>the',\n",
              " '\"<br/>there',\n",
              " '\"<br/>therefore',\n",
              " '\"<br/>these',\n",
              " '\"<br/>this',\n",
              " '\"<br/>those',\n",
              " '\"<br/>trump',\n",
              " '\"<br/>unfortunately',\n",
              " '\"<br/>united',\n",
              " '\"<br/>very',\n",
              " '\"<br/>vices',\n",
              " '\"<br/>volunteerism',\n",
              " '\"<br/>we',\n",
              " '\"<br/>we\\'re',\n",
              " '\"<br/>well',\n",
              " '\"<br/>what',\n",
              " '\"<br/>when',\n",
              " '\"<br/>whenever',\n",
              " '\"<br/>where',\n",
              " '\"<br/>which',\n",
              " '\"<br/>who',\n",
              " '\"<br/>whom',\n",
              " '\"<br/>why',\n",
              " '\"<br/>with',\n",
              " '\"<br/>ya',\n",
              " '\"<br/>yes',\n",
              " '\"<br/>you',\n",
              " '\"<br/>—from',\n",
              " '\"<br/>“we',\n",
              " '\"?',\n",
              " '\"?<br/><br/>i',\n",
              " '\"??!<br/><br/>it',\n",
              " '\"??<br/><br/>not',\n",
              " '\"????<br/>why',\n",
              " '\"[90',\n",
              " '\"actually',\n",
              " '\"american',\n",
              " '\"as',\n",
              " '\"be',\n",
              " '\"because',\n",
              " '\"con',\n",
              " '\"early',\n",
              " '\"evidence',\n",
              " '\"failed',\n",
              " '\"feeding',\n",
              " '\"give',\n",
              " '\"god',\n",
              " '\"if',\n",
              " '\"ignorance\"',\n",
              " '\"in',\n",
              " '\"it',\n",
              " '\"jewishes\"?<br/><br/>if',\n",
              " '\"ka',\n",
              " '\"law',\n",
              " '\"lean',\n",
              " '\"neanderthals\"',\n",
              " '\"no',\n",
              " '\"oh',\n",
              " '\"ok',\n",
              " '\"once',\n",
              " '\"put',\n",
              " '\"stupid\"',\n",
              " '\"that',\n",
              " '\"the',\n",
              " '\"there',\n",
              " '\"time',\n",
              " '\"trump',\n",
              " '\"trust',\n",
              " '\"uneducated\"',\n",
              " '\"we\\'ll',\n",
              " '\"we\\'ve',\n",
              " '\"western',\n",
              " '\"women',\n",
              " '\"yeah',\n",
              " '\"you',\n",
              " '\"—united',\n",
              " '&amp',\n",
              " '\\'\"<br/><br/>and',\n",
              " '\\'\"<br/><br/>doesn\\'t',\n",
              " '\\'\"<br/><br/>for',\n",
              " '\\'\"<br/><br/>i',\n",
              " '\\'\"<br/><br/>it',\n",
              " '\\'\"<br/><br/>note',\n",
              " '\\'\"<br/><br/>oh',\n",
              " '\\'\"<br/><br/>our',\n",
              " '\\'\"<br/><br/>the',\n",
              " '\\'\"<br/><br/>this',\n",
              " '\\'\"<br/><br/>what',\n",
              " '\\'\"<br/>evangelical',\n",
              " \"''\",\n",
              " \"''<br/\",\n",
              " \"''<br/><br/>germans\",\n",
              " \"'<br/><br/>'the\",\n",
              " \"'<br/><br/><a\",\n",
              " \"'<br/><br/>after\",\n",
              " \"'<br/><br/>believe\",\n",
              " \"'<br/><br/>each\",\n",
              " \"'<br/><br/>eventually\",\n",
              " \"'<br/><br/>evidently\",\n",
              " \"'<br/><br/>gun\",\n",
              " \"'<br/><br/>i\",\n",
              " \"'<br/><br/>if\",\n",
              " \"'<br/><br/>in\",\n",
              " \"'<br/><br/>just\",\n",
              " \"'<br/><br/>many\",\n",
              " \"'<br/><br/>no\",\n",
              " \"'<br/><br/>paul\",\n",
              " \"'<br/><br/>russia\",\n",
              " \"'<br/><br/>speaking\",\n",
              " \"'<br/><br/>stephen\",\n",
              " \"'<br/><br/>sweden\",\n",
              " \"'<br/><br/>the\",\n",
              " \"'<br/><br/>this\",\n",
              " \"'<br/><br/>unfortunately\",\n",
              " \"'<br/><br/>we\",\n",
              " \"'<br/><br/>well\",\n",
              " \"'<br/><br/>what\",\n",
              " \"'<br/><br/>yes\",\n",
              " \"'<br/><br/>~\",\n",
              " \"'<br/>america\",\n",
              " \"'<br/>blessings\",\n",
              " \"'<br/>everything\",\n",
              " \"'<br/>i\",\n",
              " \"'<br/>not\",\n",
              " \"'<br/>that\",\n",
              " \"'<br/>trump\",\n",
              " \"'a\",\n",
              " \"'em\",\n",
              " \"'happily\",\n",
              " \"'russia'\",\n",
              " \"'”<br/>--this\",\n",
              " \"'”<br/><br/>there\",\n",
              " \"'”<br/><br/>too\",\n",
              " '(:',\n",
              " '(ashleigh',\n",
              " '(including',\n",
              " '(met',\n",
              " '(proficiency',\n",
              " '(quality',\n",
              " '(t)he',\n",
              " ').<br/><br/>in',\n",
              " ').<br/><br/>so',\n",
              " ').<br/><br/>to',\n",
              " ').<br/><br/>yeah',\n",
              " '):',\n",
              " ')<br/>3',\n",
              " ')<br/><br/>-a',\n",
              " ')<br/><br/>not',\n",
              " ')<br/><br/>our',\n",
              " ')<br/><br/>there',\n",
              " ')<br/><br/>trump',\n",
              " ')<br/>all',\n",
              " ')<br/>another',\n",
              " ')<br/>second',\n",
              " ')<br/>the',\n",
              " '+0.9',\n",
              " '+1.10)(+0.863',\n",
              " '+10',\n",
              " '+10.0%.<br/>2',\n",
              " '+25',\n",
              " '+337',\n",
              " '+6.2',\n",
              " '+8.83',\n",
              " ',\"<br/><br/>if',\n",
              " ',.<br/><br/>the',\n",
              " ',etc',\n",
              " ',getting',\n",
              " ',i',\n",
              " ',mmmm',\n",
              " ',mr',\n",
              " '-\"a',\n",
              " '-\"neither',\n",
              " '-&gt',\n",
              " \"-'(for\",\n",
              " '-(mostly',\n",
              " '-)<br/>history',\n",
              " '-)<br/>without',\n",
              " '--',\n",
              " '---',\n",
              " '----',\n",
              " '-----',\n",
              " '------',\n",
              " '---------',\n",
              " '--------------------------------------',\n",
              " '---------------------------------------------------------------------<br/>do',\n",
              " '----<br/>the',\n",
              " '---<br/>many',\n",
              " '---a',\n",
              " '---did',\n",
              " '---dylan',\n",
              " '---if',\n",
              " '---is',\n",
              " '---just',\n",
              " '---on',\n",
              " '---our',\n",
              " '---products',\n",
              " '---yes',\n",
              " '--.<br/><br/>when',\n",
              " '--<br/',\n",
              " '--<br/>-',\n",
              " '--<br/><br/>\"the',\n",
              " '--<br/><br/><a',\n",
              " '--<br/><br/>filibuster',\n",
              " '--<br/><br/>hamlet',\n",
              " '--<br/><br/>in',\n",
              " '--<br/><br/>the',\n",
              " '--<br/><br/>when',\n",
              " '--<br/><br/>yet',\n",
              " '--<br/><br/>you',\n",
              " '--<br/>and',\n",
              " '--<br/>at',\n",
              " '--<br/>but',\n",
              " '--<br/>exactly',\n",
              " '--<br/>happy',\n",
              " '--<br/>take',\n",
              " '--a',\n",
              " '--akin',\n",
              " '--albeit',\n",
              " '--all',\n",
              " '--among',\n",
              " '--and',\n",
              " '--as',\n",
              " '--astonishingly',\n",
              " '--because',\n",
              " '--btw',\n",
              " '--but',\n",
              " '--daily',\n",
              " '--declare',\n",
              " '--deporting',\n",
              " '--depressing',\n",
              " '--dr',\n",
              " '--e.a.c',\n",
              " '--from',\n",
              " '--hamm',\n",
              " '--hangers',\n",
              " '--have',\n",
              " '--his',\n",
              " '--i',\n",
              " '--i.e',\n",
              " '--if',\n",
              " '--including',\n",
              " '--is',\n",
              " '--it',\n",
              " '--kurt',\n",
              " '--like',\n",
              " '--literally',\n",
              " '--meeting',\n",
              " '--not',\n",
              " '--one',\n",
              " '--or',\n",
              " '--president',\n",
              " '--quote--',\n",
              " '--redistribute',\n",
              " '--richard',\n",
              " '--russia',\n",
              " '--sarah',\n",
              " '--shaking',\n",
              " '--shipping',\n",
              " '--so',\n",
              " '--sound',\n",
              " '--that',\n",
              " '--the',\n",
              " '--these',\n",
              " '--they',\n",
              " '--unless',\n",
              " '--we',\n",
              " '--what',\n",
              " '--which',\n",
              " '--who',\n",
              " '--with',\n",
              " '--without',\n",
              " '--would',\n",
              " '--you',\n",
              " '-0-',\n",
              " '-0.40',\n",
              " '-1',\n",
              " '-1.37',\n",
              " '-1.37(10',\n",
              " '-100',\n",
              " '-12',\n",
              " '-120',\n",
              " '-13.7',\n",
              " '-150',\n",
              " '-16',\n",
              " '-18',\n",
              " '-1939',\n",
              " '-20',\n",
              " '-2008',\n",
              " '-2014',\n",
              " '-27',\n",
              " '-30',\n",
              " '-33',\n",
              " '-4.0',\n",
              " '-540',\n",
              " '-7',\n",
              " '-8',\n",
              " '-9',\n",
              " '-99.9',\n",
              " '-<br/',\n",
              " '-<br/><a',\n",
              " '-<br/><br/>\"how',\n",
              " '-<br/><br/>a',\n",
              " '-<br/><br/>as',\n",
              " '-<br/><br/>back',\n",
              " '-<br/><br/>in',\n",
              " '-<br/><br/>it',\n",
              " '-<br/><br/>no',\n",
              " '-<br/><br/>racism',\n",
              " '-<br/><br/>see',\n",
              " '-<br/><br/>the',\n",
              " '-<br/>and',\n",
              " '-<br/>i',\n",
              " '-<br/>it',\n",
              " '-<br/>let',\n",
              " '-<br/>lgbt',\n",
              " '-<br/>ms',\n",
              " '-<br/>mustered',\n",
              " '-<br/>no',\n",
              " '-<br/>perhaps',\n",
              " '-<br/>well',\n",
              " '-_-',\n",
              " '-a',\n",
              " '-abolished',\n",
              " '-ahem-',\n",
              " '-alas',\n",
              " '-all',\n",
              " '-all-',\n",
              " '-americans',\n",
              " '-and',\n",
              " '-and-',\n",
              " '-announced',\n",
              " '-anything-',\n",
              " \"-arland'.<br/>denied\",\n",
              " '-as',\n",
              " '-at',\n",
              " '-because',\n",
              " '-boggling',\n",
              " '-both',\n",
              " '-building',\n",
              " '-but',\n",
              " '-carefully-',\n",
              " '-cause-',\n",
              " '-ceo',\n",
              " '-clear-',\n",
              " '-competent',\n",
              " '-conservative',\n",
              " '-creating',\n",
              " '-d',\n",
              " '-drugs',\n",
              " '-e.g',\n",
              " '-economic',\n",
              " '-either',\n",
              " '-end',\n",
              " '-especially',\n",
              " '-etc',\n",
              " '-everyone',\n",
              " '-for',\n",
              " '-fox',\n",
              " '-frederick',\n",
              " '-from',\n",
              " '-get',\n",
              " '-good',\n",
              " '-greatest',\n",
              " '-greed',\n",
              " '-happen',\n",
              " '-have',\n",
              " '-hawley',\n",
              " '-he',\n",
              " '-health',\n",
              " '-her',\n",
              " '-him',\n",
              " '-how',\n",
              " '-i',\n",
              " '-if',\n",
              " '-in',\n",
              " '-in-',\n",
              " '-including',\n",
              " '-induced',\n",
              " '-information',\n",
              " '-is',\n",
              " '-islamic',\n",
              " '-ism',\n",
              " '-isms',\n",
              " '-ist',\n",
              " '-it',\n",
              " '-jared',\n",
              " '-jerusalem',\n",
              " '-just',\n",
              " '-kennedy',\n",
              " '-lack',\n",
              " '-left',\n",
              " '-legal',\n",
              " '-less',\n",
              " '-let',\n",
              " '-like',\n",
              " '-los',\n",
              " '-lsa',\n",
              " '-lunch',\n",
              " '-mahatma',\n",
              " '-many',\n",
              " '-married',\n",
              " '-mostly',\n",
              " '-my',\n",
              " '-national',\n",
              " '-none.<br/><br/>america',\n",
              " '-not',\n",
              " '-of',\n",
              " '-offer',\n",
              " '-on',\n",
              " '-only',\n",
              " '-or',\n",
              " '-owned',\n",
              " '-palestinian',\n",
              " '-partisan',\n",
              " '-payer',\n",
              " '-perfect',\n",
              " '-perhaps',\n",
              " '-profits',\n",
              " '-respect',\n",
              " '-secret',\n",
              " '-she',\n",
              " '-sister',\n",
              " '-sitting',\n",
              " '-sky',\n",
              " '-slash',\n",
              " '-starting',\n",
              " '-system-',\n",
              " '-talk-',\n",
              " '-teach',\n",
              " '-term',\n",
              " '-than',\n",
              " '-thank',\n",
              " '-that',\n",
              " '-the',\n",
              " '-they',\n",
              " '-thomas',\n",
              " '-thursday',\n",
              " '-ticket',\n",
              " '-tiered',\n",
              " '-time',\n",
              " '-to',\n",
              " '-trump?<br/><br/>the',\n",
              " '-two',\n",
              " '-unknown',\n",
              " '-walt',\n",
              " '-war',\n",
              " '-was',\n",
              " '-we',\n",
              " '-well',\n",
              " '-when',\n",
              " '-where',\n",
              " '-wiki',\n",
              " '-will',\n",
              " '-with',\n",
              " '-year',\n",
              " '-yes',\n",
              " '-yoda',\n",
              " '-you',\n",
              " '-zero-',\n",
              " '-“no',\n",
              " '-“why',\n",
              " '.\"<br/><br/>(this',\n",
              " '.\"<br/><br/>amen',\n",
              " '.\"<br/><br/>during',\n",
              " '.\"<br/><br/>great',\n",
              " '.\"<br/><br/>michelle',\n",
              " '.\"<br/>history',\n",
              " '.\"which',\n",
              " '..',\n",
              " '...',\n",
              " '....',\n",
              " '.....',\n",
              " '......',\n",
              " '.......',\n",
              " '........',\n",
              " '.........',\n",
              " ...]"
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###Using support vector machines\n",
        "\n"
      ],
      "metadata": {
        "id": "kTb2hauphKZU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Using SVC\n",
        "from sklearn.svm import SVC\n",
        "classifier = SVC(kernel='linear', C=1, random_state=0)\n",
        "\n",
        "# Create pipeline using tfidf\n",
        "pipe = Pipeline([(\"cleaner\", predictors()),\n",
        "                 ('vectorizer', tfidf_vector),\n",
        "                 ('classifier', classifier)])\n",
        "\n",
        "# model generation\n",
        "pipe.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "etbCeBgOhq_O",
        "outputId": "4003cc1a-5333-4d85-d2a7-49f4415cb51d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Pipeline(steps=[('cleaner', <__main__.predictors object at 0x7f2d20786f50>),\n",
              "                ('vectorizer',\n",
              "                 TfidfVectorizer(tokenizer=<function spacy_tokenizer at 0x7f2d2e950e60>)),\n",
              "                ('classifier', SVC(C=1, kernel='linear', random_state=0))])"
            ]
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "\n",
        "predicted = pipe.predict(X_test)\n",
        "\n",
        "# Model Accuracy\n",
        "print(\"Accuracy:\",metrics.accuracy_score(y_test, predicted))\n",
        "print(\"Precision:\",metrics.precision_score(y_test, predicted))\n",
        "print(\"Recall:\",metrics.recall_score(y_test, predicted))\n",
        "print(\"f1 score\",metrics.f1_score(y_test, predicted))\n",
        "\n",
        "Accuracy_svc = metrics.accuracy_score(y_test, predicted)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9zoB0QPMhzZJ",
        "outputId": "a7308c53-ec0a-4363-ed45-da4e748ee66c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.6430865174291432\n",
            "Precision: 0.6379325448970653\n",
            "Recall: 0.6731373636531707\n",
            "f1 score 0.6550622947870283\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Ensemble Methods\n",
        "##XGBOOST"
      ],
      "metadata": {
        "id": "5meJJv-ZjxMQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Using XGboost\n",
        "from xgboost import XGBClassifier\n",
        "classifier = XGBClassifier() #reproduce same results\n",
        "\n",
        "# Create pipeline using tfidfvector\n",
        "pipe = Pipeline([(\"cleaner\", predictors()),\n",
        "                 ('vectorizer', tfidf_vector),\n",
        "                 ('classifier', classifier)])\n",
        "\n",
        "# model generation\n",
        "pipe.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ua8EbdRqjwx_",
        "outputId": "54b5cb44-8e3e-4bc0-9e7a-134a3d3d3f5b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Pipeline(steps=[('cleaner', <__main__.predictors object at 0x7f2d2078a750>),\n",
              "                ('vectorizer',\n",
              "                 TfidfVectorizer(tokenizer=<function spacy_tokenizer at 0x7f2d2e950e60>)),\n",
              "                ('classifier', XGBClassifier())])"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "\n",
        "predicted = pipe.predict(X_test)\n",
        "\n",
        "# Model Accuracy\n",
        "print(\"Accuracy:\",metrics.accuracy_score(y_test, predicted))\n",
        "print(\"Precision:\",metrics.precision_score(y_test, predicted))\n",
        "print(\"Recall:\",metrics.recall_score(y_test, predicted))\n",
        "print(\"f1 score\",metrics.f1_score(y_test, predicted))\n",
        "\n",
        "Accuracy_xgb = metrics.accuracy_score(y_test, predicted)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SH3C_Jh7kH1p",
        "outputId": "5d90226e-4715-4287-dd85-4d198e57a4ac"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.6388513985200354\n",
            "Precision: 0.6660512597741095\n",
            "Recall: 0.5669254945461268\n",
            "f1 score 0.6125037451313292\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "###RandomForestClassifier"
      ],
      "metadata": {
        "id": "q6BR_2sUkhyl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Using RF\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "classifier = RandomForestClassifier() #reproduce same results\n",
        "\n",
        "# Create pipeline using tfidfvector\n",
        "pipe = Pipeline([(\"cleaner\", predictors()),\n",
        "                 ('vectorizer', tfidf_vector),\n",
        "                 ('classifier', classifier)])\n",
        "\n",
        "# model generation\n",
        "pipe.fit(X_train, y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rSUAWEEXkbLs",
        "outputId": "41354fca-bd8a-4cb2-aab4-a2ec3cebb71a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Pipeline(steps=[('cleaner', <__main__.predictors object at 0x7f2d1ff230d0>),\n",
              "                ('vectorizer',\n",
              "                 TfidfVectorizer(tokenizer=<function spacy_tokenizer at 0x7f2d2e950e60>)),\n",
              "                ('classifier', RandomForestClassifier())])"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import metrics\n",
        "\n",
        "predicted = pipe.predict(X_test)\n",
        "\n",
        "# Model Accuracy\n",
        "print(\"Accuracy:\",metrics.accuracy_score(y_test, predicted))\n",
        "print(\"Precision:\",metrics.precision_score(y_test, predicted))\n",
        "print(\"Recall:\",metrics.recall_score(y_test, predicted))\n",
        "print(\"f1 score\",metrics.f1_score(y_test, predicted))\n",
        "\n",
        "Accuracy_rf = metrics.accuracy_score(y_test, predicted)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VvSinzevkv75",
        "outputId": "ba2320a4-13a4-4675-bae8-f55ad0e2c573"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.6520221529296784\n",
            "Precision: 0.6758606169070428\n",
            "Recall: 0.5934553521907932\n",
            "f1 score 0.6319830683663926\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
        "import matplotlib.pyplot as plt\n",
        "#Confusion Matrix\n",
        "cm = confusion_matrix(y_true=y_test, y_pred=predicted)\n",
        "labels= y.keys()\n",
        "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)\n",
        "disp.plot()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 279
        },
        "id": "spTyjhe0wvsY",
        "outputId": "ff07ade7-32bc-4b14-cc68-ec4073bde4bf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Model comparison"
      ],
      "metadata": {
        "id": "iR7KvjIjpQks"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "Accuracy = [Accuracy_rf, Accuracy_lr, Accuracy_svc, Accuracy_xgb]\n",
        "methods = ['RandomForestClassifier','LogisticRegression', 'SupporVectorClassifier', 'XGBClassifier']\n",
        "Accuracy_pos = np.arange(len(methods))\n",
        "\n",
        "plt.figure(figsize=(8,8))\n",
        "plt.bar(Accuracy_pos, Accuracy)\n",
        "plt.ylim(0.55, 0.7)\n",
        "plt.xticks(Accuracy_pos, methods, rotation='vertical')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.xticks(fontsize=15)\n",
        "plt.yticks(fontsize=15)\n",
        "plt.title('Compare accuracy of Models', fontsize =25)\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 678
        },
        "id": "2Z8rY1iPpUJO",
        "outputId": "5587dc4e-e3c6-49f6-f98e-d02a9cc5a2fe"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 576x576 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkYAAAKVCAYAAAAqdyTtAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdfUCN9/8/8OdJpVSnfMwY2yg5IjF35aaF0cbi697Mtk+hyCZzN7aZfYYhd5uIzPgsm03kdlIkd30oudcw0g2F3FTqlO7PuX5/OF2/jnNOTlRH7fn4i+t6v6/zuq6rc87zXNf7ui6JIAgCiIiIiAhGhi6AiIiI6GXBYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgREREVeLs2bOYOHEiunfvjrZt26JNmzb49NNPDV1WtVqzZg3atGmDTz75pEqXGxcXhzZt2qBNmzZVulx6NmNDF0B1i0KhwMGDB3Hs2DFcunQJmZmZKCwshJWVFVq2bImuXbti8ODBkMlkhi6ViKrQxYsX4enpidLSUkgkEtjY2KBevXqwtrbWq39cXBz+/e9/i///4IMPsGDBggr7bNy4EcuXLxf/v2TJEgwfPvz5VoBIhcGIqszFixcxZ84c3Lx5U5xmYmICCwsLZGdn4/z58zh//jw2bNiAd999FytXroSpqanhCiaiKrN582aUlpaic+fOCAoKgo2NzQstLzw8HF9//TXMzMx0ttm1a9cLvQaRNgxGVCWOHDmCzz//HMXFxbCxscGECRPw7rvvomXLlgCeHEm6evUqIiMj8ccffyAyMhKFhYUMRkR1REJCAgDAw8PjhUNR8+bNcefOHRw6dAiDBw/W2ubixYtISkoS2xJVFY4xohd28+ZNfPHFFyguLoa9vT327t2LiRMniqEIAOrVqwcnJyfMnDkThw8fRr9+/QxXMBFVuYKCAgBAgwYNXnhZw4YNAwDs3LlTZ5uyeTx1RlWNR4zoha1atQp5eXmoX78+AgMD0bRp0wrb29jYYN26ddD2mL6HDx/iv//9L6Kjo8Vfgc2bN0fv3r0xfvx4vPLKKxp9bt++LQatw4cPQyKRICgoCCdOnEBmZiaaNGkCDw8PTJo0SfzQTkhIwIYNG3DmzBlkZWXhtddew9ChQ+Hj4wMTExON1/jkk09w+vRpTJkyBZMmTUJwcDD27duHtLQ0mJiYoH379vDy8kLv3r21rvPDhw9x4MABxMTE4ObNm3jw4AFKSkrQpEkTODs7w8vLC61bt9ba98svv8Tu3bsxbNgwLFmyBDt27MCuXbuQnJyM7OxsjXEVt2/fxubNmxETE4O7d+9CqVTitddeg6urK8aPH49mzZpVuH+0USqViIuLw+HDhxEfH4979+4hKysLFhYWaN26NTw8PDBy5Eit2668EydOYOfOnbh48SIyMzNhZmYmboNBgwahU6dOGn2Ki4uxd+9eHDhwAH///TfkcjlsbGzQvHlzvP322xgyZAjeeOMNsX35feXn56e1jjVr1iAwMBDOzs747bff1OaV7+/r64vffvsNYWFhSE1NRW5uLn799Ve4uLjU+DaZPn06wsPD4ebmhp9//lnn8m7duoX33nsPgiCItVbG1atXERwcjDNnziAjIwNmZmZo1aoVBgwYgLFjx2oc5X16cPBXX32Fr776Svz/4cOH8frrr1eqhgEDBuC///0vTp06hTt37qB58+Zq8wsKChAeHg6JRIKhQ4dizZo1z1xmZGQkdu7cib/++gtyuRxSqRROTk4YOXIk3N3dK+x7/PhxBAcH46+//oJCocAbb7yBwYMHw8vLS6/1qY73ZFJSEoKDg3H69Gncu3cPSqUS//rXv9CkSRN0794dQ4YMQatWrSq9XGIwoheUkZGBgwcPAgAGDx4MW1tbvftKJBK1/58+fRqfffYZ5HI5gP//yzMxMRGJiYnYsWMH1q1bh65du+pc5tWrVzF37lzI5XJYWlpCoVAgLS0N69evx9mzZxEcHIyTJ09i2rRpKCgogJWVFUpKSnDr1i0EBATgxo0b+PHHH3Uuv6SkBOPGjcPZs2dhbGyMBg0aQC6XIyYmBjExMTq/jFeuXIndu3cDAIyNjWFpaYnCwkKkpqYiNTUVf/75J1asWIH33ntP52sLgoDPP/8cBw8ehJGREaysrGBkpH7Q988//8TcuXNRXFwMADA1NYWRkRFSUlKQkpKCXbt2YfXq1XB1ddX5OtrcvXtX7UugQYMGMDMzQ3Z2Ns6cOYMzZ84gLCwMmzZt0jompKCgAF9++SUOHDggTrOwsIBSqURCQgISEhJw9uxZ7N27V61fWloaPv30U/E0jUQigVQqRV5eHi5evIiLFy8iJycHc+fOrdT66KOoqAiffPIJLly4AGNjY1hYWKj9zdb0NhkzZgzCw8Nx4sQJ3L17V+eXaWhoKARBQMuWLSsdioKDg+Hv7y/+aLGyskJBQQEuXLiACxcuYNeuXdi4cSNeffVVsU/Zj5WsrCwolUpYWlqqrW+9evUqVQMAmJubY8CAAdi1axd2796NKVOmqM0/ePAg8vLy0L1792eGruLiYsyZMwfh4eEAIL53Hj16hGPHjuHYsWMYNGgQ/P39tYbYshBdRiqVIikpCStWrMDx48fRuXPnCl+/Ot6TJ0+ehK+vr7hMExMTmJub4969e7h37x4uXboEExMTnT8M6BkEohcQFhYmyGQyQSaTCUePHn3u5dy9e1fo2rWrIJPJhPfff184e/asOO/MmTPCe++9J8hkMsHZ2Vm4d++eWt+0tDSxhq5duwqenp7CjRs3BEEQhIKCAuHXX38V2rZtK8hkMuHHH38UunTpIkybNk24ffu2IAiCkJeXJ/zwww/iMk6ePKlR38cffyzIZDKhS5cuQvv27YWtW7cKhYWFYu1+fn5i/6ioKI3+a9euFTZu3Chcv35dKCkpEQRBEBQKhZCQkCDMnDlTkMlkwltvvaWxboIgCHPmzBHnt2vXTti0aZOQm5sr1n7//n1BEAThxIkTgoODg9CuXTth2bJlQlpamqBUKgWlUikkJSUJU6dOFWQymdC5c2fhzp07ldo/6enpwsyZM4XDhw8Ljx49Eqfn5eUJO3fuFFxdXQWZTCYsXrxYa//PP/9ckMlkgoODg7B8+XIhPT1dnJeZmSn8+eefwrfffqvWJzc3V3j33XcFmUwmdOvWTdi2bZsgl8vF+ampqcJ///tf4ZdfflHrV7avVq9erXN9Vq9eLchkMuHjjz/WmFfW/6233hLeeustYefOnUJBQYEgCIKQlZUlrr8htsnAgQMFmUwmBAQEaF1mcXGx0LNnT0EmkwmbNm3Suf7aHDlyRPwbnjx5spCamioIgiAUFRUJu3fvFjp16iTIZDLhgw8+EEpLSzX69+3bV5DJZMLOnTsr9bplTp06Jb5+WlqacObMGUEmkwnvvPOOoFQq1dqW7aO9e/cKgiCI/bS9tr+/vyCTyYQ2bdoIq1atEnJycgRBEITs7Gy19/3y5cs1+kZFRYnzp06dKty9e1cQhCefK1u2bBEcHR3Fzy1tf0sv8p4svz2e1r9/f0Emkwnjx48Xrl+/Lk4vLCwUEhIShDVr1jz3fiBBYDCiF/Ljjz+Kb15tX+r6+vbbb8UvwAcPHmjMT09PFzp37izIZDJh/vz5avPKByMPDw+hqKhIo/8XX3whthk3bpzGB60gCMLYsWMFmUwmfP311xrzyj6IZTKZEBoaqjFfoVAIH330kVhDZU2cOFGQyWTC2rVrNeaVBSOZTCb8+uuvWvsrFAoxRISEhOh8HV9fX0Emkwnff/99pWusSHx8vBgmygJjmZiYGLH+33//Xe9llv1ttW/fXrhy5Yre/aoqGMlkMuHw4cN6v+7TqmObBAcHCzKZTHBzc9MaTg4cOCBus8zMzErVWxa6xo4dq3XZhw8fFmuOiIjQmF/VwUgQBPFvOjY2VmyXmpoqtGnTRujSpYsYWHUFo3v37gnt2rUTZDKZsHLlSq2vu2TJEkEmkwmOjo7ij4wy77//vvh3olAoNPpu3bpVfO2n/5Ze9D2pKxhlZGSI05+ul6oGB1/TC8nOzhb//bxXogiCIJ5OGDNmDBo3bqzRpmnTphgzZgwAYP/+/TqX5enpqfVKt/KHqX18fDRO45Vvc/36dZ3Lf+211zBixAiN6UZGRpg8eTIA4MaNGxUuQ5uysUnnzp3T2cba2hoffPCB1nlnzpzBzZs30bBhQ4waNUrnMoYOHQrgybiWquTk5IRGjRohPz8ff//9t9q8HTt2AABkMhnGjh2r9zLLBteOGjUK7dq1q7pi9dS6dWu88847z92/OrbJsGHDxFMmx48f15i/fft2AMC7776Lf/3rX3ov99q1a0hKSgIATJ48Wevpr3feeQcdOnQAUPF7sCqVjZ0rf1n+rl27IAgC3n///Qov5QeenHIrLS1F/fr1MXHiRK1tJk+eDFNTU5SUlIjDAoAn2yQxMVFs8/RpawAYPXo0mjRponW51fWetLCwEGt5+PChXn2ocjjGiAzu9u3bYsDq0aOHzna9evXCxo0bkZ2djbS0NLUBt2XKPrifVn7QtpOTU4VtysY4aePs7Kw1VAFA165dYWxsjNLSUly+fFljUOq1a9cQEhKCc+fO4c6dO8jPz9cYgH7//n2dr+3k5KTz9gbnz58HAOTl5eHtt9/WuYySkhIAT8bHVFZxcTF27tyJQ4cOISEhAdnZ2eLyyrt3757a/y9cuAAA6NOnj96vdefOHTx48AAA0Ldv30rXWhWeNXYEqNltAjwZ3zJw4EDs2rUL27dvVwtud+7cQUxMDIAnX9iVcfnyZQBPxr85OzvrbNezZ0/Ex8eL7avb0KFDERAQgMjISHz77bdo0KAB9uzZAwBaf6A8raxOJycnWFpaam1jbW2N9u3b4/z582rrVX6b6BrXaGRkBGdnZ+zbt09jXnW9J83MzNCjRw+cPHkS3t7eGDNmDPr06YO2bdvy9idVhMGIXkj5o0TZ2dk6fz1VJDMzU/x3Rf3Lz8vKytIajCwsLLT2Lf8LWNcHZFmb0tJSvWp4Wv369WFjY4OMjAy1dQKALVu2YNGiRVAqlQCeDCK2srISP8gKCwuRl5eH/Px8ncuv6AhAWYgoKSlBRkaGznZlCgsLn9mmvMzMTHh5eYmDoIEn69uwYUNxu5UNvi27bLtMWT2VufKm/Do8zxU7VeFZR1xqepuU+fDDD7Fr1y5ER0fj/v374t9kaGgolEolbG1tKz3oOisrCwDQsGHDCr9cy644ffrvu7o0adIEvXr1QnR0NMLDw9G8eXPcvXsX9vb26Nix4zP7l9X5rM8lbetV2W3ytOp8T37//feYPHkyrl27hnXr1mHdunUwMTGBk5MT+vXrh5EjR77wvaT+yRiM6IWUv8T877//fq5gVNclJSVh8eLFUCqVGDBgACZMmAAHBwe1D9vQ0FB88803FS6noqt7FAoFAKBjx47i6ZSqtHjxYiQkJMDGxgazZ8+Gm5ubxinP3r174969expHwXQdYavI8/Spas+6mqqmt0mZDh06wNHREVeuXEFoaCimTJkChUIhnm6q7NGil93w4cMRHR2NXbt2iUGyNty7qDrfk82aNcPu3btx8uRJHD9+HOfPn8f169fVni4QEBBQ4RF40o1jjOiFuLi4iOe7Dx069FzLaNSokfjvik4llZ9XmfETVami+oqLi8VTguXX6cCBA1AoFGjVqhV+/PFHdOjQQeMXqD6/KCtS9oX8PKfInqWkpETct99++y1GjBihEQAUCgUePXqktX/ZKcrK1Fb+1Gdl16ks0BQVFelsk5ubW6llPs0Q26S8svF2u3btglKpxPHjx3H//n2YmpqKY1Yqo+z99OjRI/EScG3KTgmW//uubv369YONjQ0uXLiAyMhIGBsbY8iQIXr1Lavz6VOZT9O2XvpuE12fCdX5ngSenMZ7++238c0332DXrl2Ii4vDihUr0KxZM+Tk5GDWrFkV1k26MRjRC3nllVfw7rvvAgDCwsKQkpKid9+yX9Gvv/66eNg3NjZWZ/uy8RM2NjZaT6PVhDNnzmi9MSXw5MniZafh2rdvL04v+9B1cHDQOoAT+P/r9rzKxsM8fPgQf/311wst62lZWVliyGjbtq3WNufOndMZRMpuUHj06FG9X7NZs2bi0cfK9AOejMMBgPT0dJ1t4uPjK7XMpxlim5Q3aNAgWFpa4s6dO/jf//733IOuy5T9vZaWluL06dM625W9P3WN06sOpqam4mNBSkpK4ObmpvVGr9qUrdfly5d1hmG5XK42FunpvqWlpTovilAqlTq3V3W+J7WxtLTE4MGDsWjRIgBPfmyVP81L+mMwohc2bdo0NGjQAIWFhfDz86vwqAoA5OTkwM/PT/ygkkgkGDhwIABg27ZtWq+0uH//PrZt2wbgyZeCody9e1e8UWN5SqUS69evBwDY29urDbwuG9OUkJCgNVQdP368wi8jfbi4uKBFixYAnjxh/Fm/FMtfTfgslpaW4qmfa9euacwvLS2t8KaYI0eOBPDkar0//vhD79ct6xcaGoqrV6/q3c/BwQHAk6t8tI3Zio2NFQc/Py9DbZMyDRo0EI+aBAUFITo6GsDzn0ZzcHCAvb29uLyy00DlHT9+HJcuXQLw5HloNemjjz7C+PHjMX78eEyaNEnvfu+99x6MjY1RVFSk827h69evR3FxMUxMTMQfecCTbVJ25+igoCBxfGB5O3fu1Hk0qrrek89aTv369cV/6/ohRhXjVqMXZmtri+XLl8PExAQ3btzAkCFDsGHDBty6dUtsU/YQ2YCAAPTv3x+RkZFqy/D19YVUKkV2djbGjRsnXtEBPPnlPW7cOPFRELouu60JVlZW+O6777B9+3bxaEB6ejpmzJiBuLg4AE+CYnlubm4AnnwJzp8/X/wAzM/PR0hICD7//PMXHihpbGyM+fPnw9jYGOfOncPHH3+M2NhYtSuk0tLSsHXrVowYMaJSX8YWFhbir19/f3/ExsaKXxIJCQmYOHEiLl++rPMZWd27dxe/SBcuXIiVK1eqfZlkZWUhNDQUX3/9tVq/8ePHo2XLliguLoaXlxe2b9+OvLw8cX5qaioCAwOxadMmtX4DBw6EkZERsrOzMWPGDPG1CgsLxbsov+j2NtQ2Ka/sdNqFCxegUCiea9B1ebNmzQLw5Mjn1KlTkZaWBuDJUZo///wTM2bMAPDkaFf//v2f+3Weh62tLebMmYM5c+bgrbfe0rtfkyZN8O9//xsAsGHDBqxevVq86lQul2PVqlXi34+Xl5faHb2BJ49hAYC4uDjMnDlT3EdFRUXYunUrFixYIB6hfFp1vScvXLiAwYMHIzg4GElJSeLfnSAIOH/+PL777jsATwaFP31lLOmHg6+pSvTv3x+bN2/GV199hVu3bmHlypVYuXIlTExMYGFhAblcrnZF1qBBg2Bubi72b9q0KdauXYtPP/0UN27cwIcffih+qZT96pdKpVi7dq1BB3iPHTsWZ8+exbx587BgwQI0aNAAOTk54vzJkydrPHepR48e8PDwwP79+7F161Zs3boVUqkUjx8/hkKhgKOjI4YPH46FCxe+UG09evRAQEAAZs+ejUuXLsHLy0vc/vn5+Wq/NCv7xfb111/jk08+wf379+Hl5QVTU1OYmJjg8ePHMDY2xqJFi7B69WqdV9UtWrQIJSUliIyMxIYNG7BhwwbxqEvZkcOyIz1lLC0tsXHjRvj6+iIxMRHz5s3Df/7zH0ilUhQVFYlXepV98ZWxtbXF5MmTsXbtWhw9ehRHjx4VH21RWlqK/v37o3Xr1ggKCqrUNngZtkl5MpkMXbp0EU/zvOig6759++Krr76Cv78/oqKiEBUVBalUioKCAvHLXCaTISAg4Lke82Eo06dPR3p6OiIiIrB27VoEBQXBysoKubm54mfSoEGD8Pnnn2v0dXd3h6+vL9avX4/w8HCEh4fD2toajx8/RmlpKbp27YouXbrgp59+0vra1fWeTEhIwJIlS7BkyRJxeXl5eeKpfEtLS6xcubJW7aeXCYMRVZkuXbogIiICBw4cwNGjRxEfH4/MzEw8fvwY1tbWsLOzQ7du3TBkyBDY2dlp9Hd2dkZ4eDh++eUXHD9+HHfu3IFEIkGrVq3Eh8hqu/ljTTIxMUFwcDB++eUXhIWFIS0tDVZWVmjfvj3GjRun8yGyK1asQMeOHbFz506kpKRAoVBAJpPh/fffh5eXF8LCwqqkvv79++PQoUP4448/EB0djVu3biE3Nxfm5uaws7ODk5MT+vTpIx7F0lf79u0RGhqKwMBAnDp1Cnl5ebCwsICbmxvGjx+PDh06YPXq1Tr7m5ubY82aNTh27Bh27NiBS5cu4dGjR7CwsECbNm3g7OyM//u//9Po98Ybb2D37t3YsWMHIiIikJCQgMePH6Nhw4ZwcHCAm5ub1oG4U6dORYsWLfDHH38gISEBCoUCDg4OGDVqFD744AO1Z189L0Ntk/IGDBiAc+fOPfeg66d5eXmhW7duGg+RdXR0xMCBA7U+RPZlZ2pqilWrVmHgwIHYsWMHLl++LB59bt++PUaPHl3hQ2SnT5+OTp064ZdffsHly5dRXFwMOzs7DB48GOPGjRNPoetS1e9JJycnrFq1CnFxcYiPj8eDBw+QnZ0NU1NTtG7dGr169cK///1vXiH8AiSCrpGkNSgxMRELFy7ExYsXYWVlhVGjRmHKlCkVpt2nH+xX3owZM9TOQ0dFRSEgIAA3b97EG2+8gSlTpuD999+v8vWgukufJ7YT1TRfX18cPXoUgwYNwsqVKw1dDlGdYPAjRjk5OfDy8oK9vT3WrVuH1NRULF26FEqlUjy/q82oUaM07iYaFRWFn3/+WS15l50vHzt2LObOnYvjx49jxowZkEqllX7COBHRyyItLU18LMiHH35o4GqI6g6DB6OQkBAUFRUhMDAQlpaW6NWrF/Ly8hAYGAgfHx+ddylu2rSpxh1H161bBzs7O7XLZ4OCgtC1a1fx5nndu3dHYmIi1q5dy2BERLVSXl4evvvuOyiVSnTs2FHnIyuIqPIMflVadHQ0XF1d1QKQh4cHCgsLK3UJ86NHjxATE6N2GWlxcTHi4uLES8HLL//ixYsvfJM3IqKatHTpUvTt2xfdu3fHiRMnYGxsXOGVa0RUeQYPRsnJyRoDcZs1awZzc3MkJyfrvZzIyEiUlJSo3eMmNTUVJSUlGsu3s7ODUqms1M0IiYgM7dGjR7h79y5MTEzQqVMnbNy4sVKXrxPRsxn8VJpcLoeVlZXGdKlUWuFTzp+2f/9+ODo6omXLluK0ssuon77PhLW1tfjaRPr47bffDF0CEfz9/eHv72/oMojqNIMfMaoKDx48wJkzZ2r8bqxERERUtxj8iJFUKlW7m20ZuVyu846iT4uIiIAgCBqX4JcdGXp6LJGuI0nP8ujRYyiVBr+7wUunUSNLZGZq7kOqG7h/6zbu37qL+1Y7IyMJGja00Dnf4MHIzs5OYyxReno6CgoKtN4EUJvw8HB06dIFr732mtr0N998EyYmJkhOToazs7M4PTk5GUZGRrC1ta1UrUqlwGCkA7dL3cb9W7dx/9Zd3LeVZ/BTaW5ubjhx4oTaUaPw8HCYmZmphRldbt++jYsXL2o9jWZqagoXFxccOHBAbXpERATeeustrWObiIiI6J/L4MFozJgxMDU1hZ+fH2JiYrBt2zYEBgbCy8tL7RJ+d3d3rZelhoeHw9jYGAMGDNC6/MmTJ+P06dNYtGgR4uLisGzZMhw/fhyfffZZta0TERER1U4GD0bW1tYIDg6GQqGAr68v1qxZA09PT0ydOlWtnUKhEB/4V97+/fvRvXt3/Otf/9K6/K5du2L16tWIjY3FhAkTcOTIEaxcuZI3dyQiIiINL8Wz0mqLzMw8nq/VonFjKzx8yJtl1lXcv3Ub92/dxX2rnZGRBI0aaX+qBvASHDEiIiIielkwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqTAYEREREakwGBERERGpMBgRERERqbwUwSgxMRGenp7o2LEjXF1dERAQAIVCoVffyMhIjBgxAh06dICLiwsmTJiA/Px8cX5xcTECAwPh7u6ODh06wN3dHatXr0ZxcXF1rQ4RERHVUsaGLiAnJwdeXl6wt7fHunXrkJqaiqVLl0KpVGL69OkV9g0NDcWCBQvg7e2N2bNnQy6X49SpU2qhauXKlQgJCcG0adPQtm1bXL16FatWrYJcLsc333xT3atHREREtYjBg1FISAiKiooQGBgIS0tL9OrVC3l5eQgMDISPjw8sLS219svKysLixYsxb948jB49Wpzu7u6u1i4sLAwffvghxo0bBwDo3r077t+/j3379jEYERERkRqDn0qLjo6Gq6urWgDy8PBAYWEhTp8+rbNfREQEAGDo0KEVLr+0tFQjXFlZWUEQhBeomoiIiOoigwej5ORk2NnZqU1r1qwZzM3NkZycrLNffHw8bG1tsWPHDri5ucHR0RGjRo3C+fPn1dqNHDkS27Ztw7lz5/D48WOcPXsWISEh+Oijj6plfYiIiKj2MvipNLlcDisrK43pUqkUcrlcZ7+MjAykpKQgKCgIX3zxBWxsbLBx40Z4e3sjMjISr7zyCgBg1qxZKCoqwtixY8W+Y8eOxZQpU6p+ZYiIiKhWM3gwel6CICA/Px8BAQFwc3MDAHTu3Bl9+/bFli1bMG3aNADAxo0b8eeff6OLSXcAACAASURBVGLevHlo06YNrl27hoCAANjY2ODzzz+v1Gs2aqR9vBMBjRtrhluqO7h/6zbu37qL+7byDB6MpFIp8vLyNKbL5XJIpdIK+0kkEri4uIjTLC0t4ejoiKSkJABPBmgHBATg22+/FQdod+vWDSYmJli4cCE+/vhjNGrUSO9aMzPzoFRybNLTGje2wsOHuYYug6oJ92/dxv1bd3HfamdkJKnwQIfBxxjZ2dlpjCVKT09HQUGBxtij8lq1agVBEDQGUQuCAIlEAgC4ffs2SkpK4ODgoNamXbt2KC0txZ07d6poLYiIiKguMHgwcnNzw4kTJ9SOGoWHh8PMzAzOzs46+/Xp0wcAEBcXJ07Lzc3FlStXxCDUrFkzAMDVq1fV+l6+fBkA8Prrr1fJOhAREVHdUO+77777zpAFtG7dGtu2bUNcXBxeffVVxMTE4IcffoCnpyd69+4ttnN3d8e1a9fQr18/AECTJk3w999/Y+vWrWjYsCHu37+PhQsXIjs7G8uWLYOZmRkaNGiAa9euITQ0FPXr10dhYSGioqKwatUqvPPOOxg+fHilai0oKAav8tdkYVEf+fm8k3hdxf1bt3H/1l3ct9pJJBI0aGCqc77BxxhZW1sjODgYCxYsgK+vL6RSKTw9PeHn56fWTqFQQKlUqk1bvnw5li1bBn9/fxQUFKBz587YvHkzrK2txTZLly7F2rVr8dtvv+HBgwdo0qQJPvjgA3z66ac1sn5ERERUe0gE3ulQbxx8rR0H+NVt3L91G/dv3cV9q91LP/iaiIiI6GXBYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpMJgRERERKTCYERERESkwmBEREREpPJSBKPExER4enqiY8eOcHV1RUBAABQKhV59IyMjMWLECHTo0AEuLi6YMGEC8vPz1do8evQI3377LXr16oUOHTpgwIAB2LNnT3WsChEREdVixoYuICcnB15eXrC3t8e6deuQmpqKpUuXQqlUYvr06RX2DQ0NxYIFC+Dt7Y3Zs2dDLpfj1KlTaqEqLy8PH3/8MRo0aIBvvvkGDRs2RFJSEkpKSqp71YiIiKiWMXgwCgkJQVFREQIDA2FpaYlevXohLy8PgYGB8PHxgaWlpdZ+WVlZWLx4MebNm4fRo0eL093d3dXarV+/HsXFxdi5cyfMzMwAAN27d6++FSIiIqJay+Cn0qKjo+Hq6qoWgDw8PFBYWIjTp0/r7BcREQEAGDp0aIXL37VrF0aOHCmGIiIiIiJdDB6MkpOTYWdnpzatWbNmMDc3R3Jyss5+8fHxsLW1xY4dO+Dm5gZHR0eMGjUK58+fF9ukpaUhMzMTVlZW8PHxQfv27dG9e3csWbIExcXF1bZOREREVDsZPBjJ5XJYWVlpTJdKpZDL5Tr7ZWRkICUlBUFBQZg1axaCgoJgbm4Ob29vZGRkiG0AYPny5WjSpAl+/vln+Pr6YuvWrVi1alX1rBARERHVWgYfY/S8BEFAfn4+AgIC4ObmBgDo3Lkz+vbtiy1btmDatGkQBAEA0Lp1a3z//fcAgB49eiAvLw8//fQT/Pz8YG5urvdrNmqkfbwTAY0ba4Zbqju4f+s27t+6i/u28gwejKRSKfLy8jSmy+VySKXSCvtJJBK4uLiI0ywtLeHo6IikpCQAgLW1NQCotQGeDL5es2YNUlNT0aZNG71rzczMg1Ip6N3+n6JxYys8fJhr6DKomnD/1m3c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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Referrences:\n",
        "\n",
        "* https://spacy.io/usage/linguistic-features\n",
        "* https://www.kaggle.com/code/aashita/predicting-nyt-s-pick/notebook\n",
        "* https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/\n",
        "* https://help.nytimes.com/hc/en-us/articles/115014792387-Comments#:~:text=What%20are%20NYT%20Picks%3F,hand%20knowledge%20of%20an%20issue.\n",
        "\n"
      ],
      "metadata": {
        "id": "WwxZ_xiUPZLn"
      }
    }
  ]
}
