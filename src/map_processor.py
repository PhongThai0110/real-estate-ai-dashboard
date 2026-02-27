import pandas as pd
import geopandas as gpd
import json
import os
import glob
from shapely.geometry import shape, Point
from tqdm import tqdm

# ======================================================
# 1. HÀM ĐỌC JSON (GIỮ NGUYÊN)
# ======================================================
def process_custom_json(file_path):
    try:
        # Đọc JSON bắt buộc dùng utf-8
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        province_name = data.get('name', 'Unknown')
        districts_data = data.get('level2s', [])
        
        if not districts_data:
            return None

        parsed_districts = []
        for d in districts_data:
            district_name = d.get('name')
            geo_type = d.get('type')
            coords = d.get('coordinates')
            
            geo_json_struct = {'type': geo_type, 'coordinates': coords}
            try:
                geometry = shape(geo_json_struct)
                if geometry.is_valid:
                    parsed_districts.append({'district_name': district_name, 'geometry': geometry})
                else:
                    clean_geo = geometry.buffer(0)
                    if clean_geo.is_valid:
                        parsed_districts.append({'district_name': district_name, 'geometry': clean_geo})
            except:
                continue

        if parsed_districts:
            gdf = gpd.GeoDataFrame(parsed_districts)
            gdf.set_crs(epsg=4326, inplace=True)
            return gdf
        return None
    except Exception as e:
        print(f"Lỗi đọc JSON {file_path}: {e}")
        return None

# ======================================================
# 2. HÀM TẠO BẢN ĐỒ TỔNG
# ======================================================
def load_master_map(geojson_folder):
    print(f"🔄 Đang tải tất cả bản đồ từ: {geojson_folder}...")
    json_files = glob.glob(os.path.join(geojson_folder, "*.json"))
    
    if not json_files:
        print("❌ Không tìm thấy file .json nào!")
        return None
    
    all_gdfs = []
    for file_path in tqdm(json_files, desc="Loading Maps"):
        gdf = process_custom_json(file_path)
        if gdf is not None:
            all_gdfs.append(gdf)
            
    if all_gdfs:
        master_gdf = pd.concat(all_gdfs, ignore_index=True)
        print(f"✅ Đã tạo Master Map với {len(master_gdf)} quận/huyện.")
        return master_gdf
    else:
        return None

# ======================================================
# 3. HÀM XỬ LÝ FILE DATA (SỬA LỖI FONT Ở ĐÂY)
# ======================================================
def map_data_with_master(df, master_gdf):
    # ... (Giữ nguyên logic cũ) ...
    if 'lat' not in df.columns or 'lon' not in df.columns:
        print("❌ Lỗi: Data thiếu cột 'lat' hoặc 'lon'")
        return df

    # Tạo Geometry
    try:
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        gdf_data = gpd.GeoDataFrame(df, geometry=geometry)
        gdf_data.set_crs(epsg=4326, inplace=True)
    except Exception as e:
        return df

    # Spatial Join
    joined = gpd.sjoin(gdf_data, master_gdf[['district_name', 'geometry']], how='left', predicate='within')
    
    if 'index_right' in joined.columns:
        joined = joined.drop(columns=['index_right'])
    
    if 'district_name' in joined.columns:
        joined.rename(columns={'district_name': 'district_mapped'}, inplace=True)
    
    final_df = pd.DataFrame(joined.drop(columns='geometry'))
    return final_df

# ======================================================
# 4. CHẠY THỰC TẾ (QUAN TRỌNG NHẤT LÀ PHẦN GHI FILE)
# ======================================================
if __name__ == "__main__":
    GEOJSON_FOLDER = "data/geojson"
    DATA_FILES = [
        "data/data_apartment_final.csv",
        "data/data_nha_hcm_final.csv",
        "data/data_nha_hn_final.csv",
        "data/data_land_all_final.csv",
        "data/data_villa_vip_final.csv"
    ]

    master_map = load_master_map(GEOJSON_FOLDER)
    
    if master_map is not None:
        for file_path in DATA_FILES:
            print(f"\n📂 Đang xử lý file: {file_path}")
            
            if not os.path.exists(file_path):
                continue
                
            try:
                # --- SỬA LỖI 1: Đọc file với encoding utf-8-sig ---
                # utf-8-sig giúp đọc đúng tiếng Việt trên Windows
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                except:
                    # Nếu lỗi thì thử utf-8 thường
                    df = pd.read_csv(file_path, encoding='utf-8')
                
                # Map dữ liệu (Ghi đè lại cột district_mapped cũ bị lỗi)
                df_mapped = map_data_with_master(df, master_map)
                
                if 'district_mapped' in df_mapped.columns:
                    success = df_mapped['district_mapped'].notna().sum()
                    print(f"   ✅ Map lại thành công: {success}/{len(df_mapped)} dòng.")
                    
                    # --- SỬA LỖI 2: Ghi file với encoding utf-8-sig ---
                    # Đây là bước quan trọng nhất để Excel hiển thị đúng tiếng Việt
                    df_mapped.to_csv(file_path, index=False, encoding='utf-8-sig')
                    print(f"   💾 Đã lưu file (UTF-8-SIG): {file_path}")
                else:
                    print("   ⚠️ Không tạo được cột district_mapped.")
                    
            except Exception as e:
                print(f"   ❌ Lỗi xử lý file {file_path}: {e}")