var board = null;

document.addEventListener("DOMContentLoaded", function(event) {
  var cfg = {
    draggable: true,
    dropOffBoard: 'snapback', // this is the default
    position: 'start',
    onDrop: getMove
  };

  board = ChessBoard('board', cfg);
});

var make_move = false;
function getMove(oldLocation, newLocation, piece, n_pos, o_pos) {
  if (!make_move) {
    var move = oldLocation + newLocation;
    if ((piece == 'wP' || piece == 'bP') && (newLocation.slice(-1) == '1' || newLocation.slice(-1) == '8')) {
      move += 'q'
    }

    var old_fen = ChessBoard.objToFen(o_pos);
    console.log('Move to make:', move)
    console.log('Old board:', old_fen)
    $.ajax({
      url: "http://0.0.0.0:5557/getmove",
      type: "POST",
      crossDomain: true,
      data: JSON.stringify({move: move, fen: old_fen}),
      dataType: "json",
      success: function (response) {
        var fen = response[0].fen
        console.log('New board:', fen)
        $('#latest-move').html(response[0].explination)
        make_move = true;
        board.position(fen, false)
        make_move = false;
      },
      error: function (xhr, status) {
        alert("error");
      }
    });
  }
}