% 经纬高转局部xyz
function avp_xyz = llh2xyz_subfun(avp_llh)
    pos = avp_llh(7:9);
    lat0 = 34.13801;
    lon0 = -118.12528;
    h0 = 2.0470;
    pos0 = [lat0/180*pi,lon0/180*pi,h0]';
    pos = pos2dxyz(pos,pos0);
    avp_xyz(7:8) = pos(1:2);
    avp_xyz(9) = avp_llh(9);
    avp_xyz(1:6) = avp_llh(1:6);
%     [xEast2, yNorth2, zUp2] = geodetic2enu(aps_att(i,4)*180/pi,aps_att(i,5)*180/pi,aps_att(i,6),lat0,lon0,h0,wgs84, 'degrees');
%     aps_att(i,4:5) = [xEast2,yNorth2];
end