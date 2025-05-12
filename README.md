# FramePack Storyboard

為FramePack新增Storyboard功能來更好的產出所想要的影片，目前利用加入最後幀功能的FramePack來實現

在Storyboard中可以隨意在120秒的時間內指定10個節點，每個節點指定影片幀，每個節點間下達文字prompt

# 下載方式：
下載FramePack，並執行一次run.bat確保原本的FramePack正常執行

然後將本專案的framepack_start_end.py、storyboard.html、storyboard_server.py放入webui資料夾

# 使用方式：
將run.bat中執行的python檔案換成storyboard_server.py

# 介面呈現：
支持縮放時間軸、顯示每個時間段上傳的圖片

![螢幕擷取畫面 2025-05-12 223325](https://github.com/user-attachments/assets/4924e4cc-94bd-47ba-8ff2-104053576a94)
![螢幕擷取畫面 2025-05-12 223422](https://github.com/user-attachments/assets/ee57df9b-11d5-4ff3-a7eb-2328f9c8ff39)
![螢幕擷取畫面 2025-05-12 223403](https://github.com/user-attachments/assets/23800272-51f4-4279-aa48-05614d9a8e08)
![螢幕擷取畫面 2025-05-12 223434](https://github.com/user-attachments/assets/b8c4cea1-19f4-433c-83f0-615061c01d17)

如果Process Storyboard成功，會產生json檔案到storyboard_outputs\json_output資料夾內，最終成品影片在storyboard_outputs\video_output
![image](https://github.com/user-attachments/assets/284c66ad-edb0-4949-a5f6-6e4cc05240bd)

json檔案即Storyboard內容，可以重複使用，未來可以用下方的Process More Storyboards按鈕直接執行storyboard_outputs\json_output下的Storyboard
