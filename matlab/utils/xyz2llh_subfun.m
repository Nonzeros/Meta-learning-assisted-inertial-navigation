function avp_llh = xyz2llh_subfun(avp_xyz)
avp_llh = avp_xyz;
% 东北天转经纬高
lat0 = 34.13801;
lon0 = -118.12528;
h0 = 2.0470;
wgs84 = wgs84Ellipsoid;

[lat,lon,h] = enu2geodetic(avp_xyz(7),avp_xyz(8),avp_xyz(9),lat0,lon0,h0,wgs84); % 出来的经纬度是角度
p = [lat*pi/180 ,lon*pi/180]; % 经纬度弧度表示
avp_llh(7:8) = p;
end


% 经纬高转局部xyz
function avp_xyz = llh2xyz_subfun(avp_llh)
    pos = avp_llh(7:9);
    lat0 = 34.13801;
    lon0 = -118.12528;
    h0 = 2.0470;
    pos0 = [lat0/180*pi,lon0/180*pi,h0]';
    pos = pos2dxyz(pos,pos0);
    avp_xyz(7:9) = pos;
    avp_xyz(1:6) = avp_llh(1:6);
%     [xEast2, yNorth2, zUp2] = geodetic2enu(aps_att(i,4)*180/pi,aps_att(i,5)*180/pi,aps_att(i,6),lat0,lon0,h0,wgs84, 'degrees');
%     aps_att(i,4:5) = [xEast2,yNorth2];
end
