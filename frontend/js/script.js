var board = null;

document.addEventListener("DOMContentLoaded", function(event) {
  var cfg = {
    draggable: true,
    dropOffBoard: 'snapback', // this is the default
    position: 'start',
    onChange: getMove
  };

  board = ChessBoard('board', cfg);
});

var make_move = false;
function getMove(oldPos, newPos) {
  var fen = ChessBoard.objToFen(newPos); //  + ' b'
  console.log(fen)
  if (!make_move) {
    $.ajax({
      url: "http://0.0.0.0:5557/getmove",
      type: "POST",
      crossDomain: true,
      data: JSON.stringify({fen: fen}),
      dataType: "json",
      success: function (response) {
        var move = response[0].move.slice(0,2) + '-' + response[0].move.slice(2, 4)

        console.log('Got move:', response[0].move)
        console.log(response[0].explination)
        $('#latest-move').html(response[0].explination)
        make_move = true;
        board.move(move)
        make_move = false;
      },
      error: function (xhr, status) {
        alert("error");
      }
    });
  }
}