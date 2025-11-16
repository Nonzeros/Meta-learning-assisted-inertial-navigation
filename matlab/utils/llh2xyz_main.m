% 经纬高转局部xyz
function ukf_avps = llh2xyz_main(ukf_avps_llh)
for i = 1:length(ukf_avps_llh)
    pos = ukf_avps_llh(i,7:10);
    pos0 = [lat0/180*pi,lon0/180*pi,h0]';
    pos = pos2dxyz(pos,pos0);
    avp(i,7:10) = pos;
%     [xEast2, yNorth2, zUp2] = geodetic2enu(aps_att(i,4)*180/pi,aps_att(i,5)*180/pi,aps_att(i,6),lat0,lon0,h0,wgs84, 'degrees');
%     aps_att(i,4:5) = [xEast2,yNorth2];
end