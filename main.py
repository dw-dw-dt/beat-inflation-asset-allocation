from utils import load_jp_cpi_df


if __name__ == "__main__":
    cpi_df = load_jp_cpi_df()
    print(cpi_df.head())
    
