import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calib, undistort
from threshold import gradient_combine, hls_combine, comb_result
from finding_lines import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map
from skimage import exposure
input_type = 'video' #'video' # 'image'
input_name = 'project_video.mp4' #'test_images/straight_lines1.jpg' # 'challenge_video.mp4'

import os
import shutil
from os.path import join, getsize

path = '/home/test/test/Advanced-lane-finding/'
dirname = 'output'
count = 1

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

start_frame = 0   # 시작 시, frame

left_line = Line()
right_line = Line()

th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()

out = None

def videosave(dst_path, fourcc, fps, frame_size):
    # 영상 저장하기
    out = cv2.VideoWriter(dst_path, fourcc, fps, frame_size, isColor=True)
    return out

if __name__ == '__main__':
    if input_type == 'video':
        cap = cv2.VideoCapture(input_name)

        # 해당 디렉토리 여부 및 없으면 생성
        dst_dir = os.path.join(path, dirname)
        if (os.path.isdir(dst_dir) == False):
            os.mkdir(dst_dir)

        total_frame = 0

        while (cap.isOpened()):
            _, frame = cap.read()
            cv2.imshow('frame', frame)
            # Correcting for Distortion
            undist_img = undistort(frame, mtx, dist)
            # resize video
            undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
            rows, cols = undist_img.shape[:2]
            # rows, cols, _ = undist_img.shape

            combined_gradient = gradient_combine(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
            # cv2.imshow('gradient combined image', combined_gradient)

            combined_hls = hls_combine(undist_img, th_h, th_l, th_s)
            # cv2.imshow('HLS combined image', combined_hls)

            combined_result = comb_result(combined_gradient, combined_hls)

            c_rows, c_cols = combined_result.shape[:2]
            s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
            s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

            src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
            dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

            warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
            # cv2.imshow('warp', warp_img)

            searching_img = find_LR_lines(warp_img, left_line, right_line)
            # cv2.imshow('LR searching', searching_img)

            w_comb_result, w_color_result = draw_lane(searching_img, left_line, right_line)
            # cv2.imshow('w_comb_result', w_comb_result)

            # Drawing the lines back down onto the road
            color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
            lane_color = np.zeros_like(undist_img)
            lane_color[220:rows - 12, 0:cols] = color_result

            # Combine the result with the original image
            result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
            # cv2.imshow('result', result.astype(np.uint8))

            info, info2 = np.zeros_like(result), np.zeros_like(result)
            info[5:110, 5:190] = (255, 255, 255)
            info2[5:110, cols - 111:cols - 6] = (255, 255, 255)
            info = cv2.addWeighted(result, 1, info, 0.2, 0)
            info2 = cv2.addWeighted(info, 1, info2, 0.2, 0)
            road_map = print_road_map(w_color_result, left_line, right_line)
            info2[10:105, cols - 106:cols - 11] = road_map
            info2 = print_road_status(info2, left_line, right_line)
            cv2.imshow('road info', info2)

            fps = cap.get(cv2.CAP_PROP_FPS)  # 초 당 프레임 수
            #print('fps : {}'.format(fps))

            total_frame = total_frame + 1  # 총 재생된 frame 수
            #print("총 재생된 frame : {}".format(total_frame))

            # 프레임 사이즈 확인하기
            #print('rows : {}'.format(rows))
            #print('cols : {}'.format(cols))

            # frame_size = (cols, rows)
            frame_size = info2.shape[:2][::-1]
            #print('frame_size : {}'.format(frame_size))

            # 파일 이름 설정하기
            dst_name = 'output%i' % count + '.mp4'
            dst_path = os.path.join(dst_dir, dst_name)

            # 영상 저장하기
            if out == None:  # VideoWriter가 없다면
                out = videosave(dst_path, fourcc, fps, frame_size)
                print("VideoWriter 생성됨")

            out.write(info2)  # VideoWriter에 write 하기
            #print('10초 여부 확인 : {}'.format(int(total_frame % (fps * 10))))

            if (int(total_frame % (fps * 10)) == 0):
                # 디스크 용량 확인하기
                diskLabel = '/'
                total, used, free = shutil.disk_usage(diskLabel)
                print('{} total : {} G'.format(diskLabel, ((total//1024)//1024)//1024))
                print('{} used : {} G'.format(diskLabel, ((used//1024)//1024)//1024))
                print('{} free : {} G'.format(diskLabel, ((free//1024)//1024)//1024))

                os.system('du -sh /home/test/test/Advanced-lane-finding/output/*')

                for root, dirs, files in os.walk('.'):
                    result = "%s : %.f MB in %d files." % (os.path.abspath(root),
                                                           sum([getsize(join(root, name)) for name in files]) / (
                                                                   1024.0 * 1024.0), len(files))
                print(result)

                # 10초마다 저장하기
                #print("out : {}, type : {}".format(out, type(out)))
                if out.isOpened():
                    print('Avlie')

                out.release()

                if out.isOpened():
                    print('still Avlie!!!!!')

                out = None
                #print("out : {}, type : {}".format(out, type(out)))

                # 디렉토리의 파일 리스트 가져오기
                file_list = os.listdir(dst_dir)
                print(file_list)

                count = count + 1  # 파일이름 변경하기 위해

                # 특정 용량이 넘으면 가장 오래된 파일 삭제하기


            # out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.waitKey(0)
            # if cv2.waitKey(1) & 0xFF == ord('r'):
            #    cv2.imwrite('check1.jpg', undist_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
